# Análise das falhas da bateria de testes

Resultado final (execução de 20/11/2026 16:44, LLM `groq` / `openai/gpt-oss-120b`):
**roteamento correto 14/14 (100%)**, desfecho correto 14/14 (100%), guarda anti-alucinação acionada em 0 casos.
Tempo por caso: média de 10,6 s (mínimo 3,2 s, máximo 28,5 s). Detalhes em [`resultados_bateria.md`](resultados_bateria.md).

O 100% só veio depois de várias execuções com falha. Esta análise registra o que deu errado em cada uma, a causa e a correção feita.
O `rodar_bateria.py` sobrescreve o `resultados_bateria.md` a cada execução, por isso as rodadas com falha foram copiadas para arquivos separados.

## Histórico das execuções (20/11/2026)

| Execução | Horário | Rota correta | Desfecho correto | O que aconteceu | Evidência |
|---|---|---|---|---|---|
| A | 14:42 | – | – | Teste manual: modelo `llama-3.3-70b-versatile` descontinuado no Groq (erro 404) | `logs/execucoes.jsonl` |
| B | 14:44–14:45 | 2/14 | 2/14 | `GROQ_API_KEY` vazia: o `.env` foi sobrescrito pelo `.env.example` | `logs/execucoes.jsonl` |
| 1 | 14:49 | 14/14 | 14/14 | T03: guarda acionada por engano (falha 1) | registrada abaixo (arquivo sobrescrito) |
| 2 | 16:27 | 14/14 | 13/14 | T03: LLM não extraiu `PhoneService` (falha 2) | [`resultados_bateria_execucao2.md`](resultados_bateria_execucao2.md) |
| 3 | 16:33 | 9/14 | 8/14 | Queda de conexão e limite de tokens do Groq (falha 4) | `logs/execucoes.jsonl` |
| 4 | 16:38 | 14/14 | 14/14 | T05: guarda acionada por engano (falha 3) | [`resultados_bateria_execucao3.md`](resultados_bateria_execucao3.md) |
| 5 (final) | 16:44 | 14/14 | 14/14 | Nenhuma falha | [`resultados_bateria.md`](resultados_bateria.md) |

## Falha 1 — Guarda lia "0.074" como milhar (T03, execução 1)

- **Sintoma:** a resposta do LLM foi descartada com a mensagem "citou números sem fonte: ['7,4']".
- **Causa:** o número estava certo. 113 dos 1.526 clientes sem internet cancelaram, e a ferramenta devolve `0.074`. A função que lê números (`_converterTextoEmNumero`) foi feita para texto em português e trata "0.074" como milhar, igual a "1.500". Assim, a fonte virava 74 e o "7,4" do LLM não batia com nada.
- **Tipo:** bug da guarda (falso positivo), não alucinação do LLM. A resposta final continuou correta porque o texto foi substituído pela resposta montada por código.
- **Correção:** em `encontrarNumerosSemFonte` (`agente/nos_auxiliares.py`), os números da fonte que não têm vírgula passam a ser aceitos também na leitura decimal. A leitura do texto do LLM não mudou.
- **Resultado:** nas execuções seguintes o T03 cita "7,40 %" sem acionar a guarda.

## Falha 2 — LLM não extraiu `PhoneService` (T03, execução 2)

- **Sintoma:** desfecho `pedido_correcao` em vez de `previsao`, com a mensagem "Faltam 1 campo(s): PhoneService".
- **Causa:** a entrada diz "tem telefone com múltiplas linhas". Na execução 1 o LLM extraiu `PhoneService = Yes`; na execução 2 não extraiu. É variação do LLM mesmo com temperatura 0.
- **Comportamento do agente:** seguro. Pediu o campo que faltava em vez de inventar um valor.
- **Correção:** nova regra de coerência em `_preencherPorCoerencia` (`agente/esquema.py`). Se `MultipleLines` é "Yes" ou "No", `PhoneService` só pode ser "Yes", porque no dataset quem não tem telefone tem `MultipleLines = "No phone service"`. Se o usuário informar `PhoneService = No` junto com múltiplas linhas, a validação continua acusando a incoerência.
- **Resultado:** nas execuções 4 e 5 o T03 mostra "Preenchidos por coerência: ..., PhoneService".

## Falha 3 — Guarda quebrava "3 875" em dois números (T05, execução 4)

- **Sintoma:** a resposta do LLM foi descartada ("citou números sem fonte: ['875']") e substituída pelo resultado da consulta em JSON cru.
- **Causa:** o LLM escreveu a quantidade de clientes com espaço como separador de milhar ("3 875"). A expressão regular dos números separava "3" e "875", e "875" não existe na fonte (o valor é 3875).
- **Tipo:** bug da guarda (falso positivo).
- **Correção:** antes da verificação, `encontrarNumerosSemFonte` junta dígitos separados por espaço (inclusive espaços não separáveis) quando vêm seguidos de exatamente 3 dígitos. Um número inventado no mesmo formato, como "4 120", continua sendo barrado.
- **Resultado:** na execução 5 a guarda não foi acionada em nenhum caso.

## Falha 4 — Erros do LLM apareciam como "fora do escopo" (execuções A, B e 3)

- **Sintoma:** perguntas válidas recebiam a resposta "Essa solicitação está fora do meu escopo".
- **Causa:** o roteador captura qualquer exceção do LLM e manda o fluxo para `fora_escopo`. As exceções foram:
  - modelo descontinuado no Groq (404);
  - chave ausente;
  - queda de conexão (T01 a T04 da execução 3);
  - limite de 8.000 tokens por minuto do plano gratuito do Groq (429, T11 da execução 3);
  - requisição recusada durante a extração (T14 da execução 3), que virou pedido de correção.
- **Impacto:** nenhuma previsão foi inventada, mas a mensagem engana o usuário e contraria o princípio "se uma ferramenta falhar, o agente diz que falhou". O motivo real fica registrado em `justificativaRota` e no log.
- **O que foi feito:**
  - modelo trocado para `openai/gpt-oss-120b`;
  - criado o `.env.example`;
  - a bateria foi repetida com a conexão estável.
- **Melhoria possível (não implementada):** uma resposta própria para erro do LLM ("não consegui consultar o modelo de linguagem agora") e uma pausa entre os casos para não estourar o limite de tokens.

## Erro do modelo, não do agente (T01 e T14)

- O cliente `7590-VHVEG` tem `Churn = No` na base, mas o modelo prevê "Cancela" com probabilidade acima de 99,9%. É um falso positivo com confiança extrema, e acontece mesmo com o cliente no conjunto de treino.
- Isso é coerente com o [cartão do modelo](../docs/cartao_modelo.md): precisão de 0,4572 e probabilidades mal calibradas do Naive Bayes. O modelo foi escolhido pelo menor custo total (FN = R$ 10.000, FP = R$ 100), o que favorece o recall e aceita muitos falsos positivos.
- O agente se comporta corretamente: mostra a previsão do modelo, informa o `Churn` real da base e avisa que o cliente estava no treino. No T14, mantém essa previsão mesmo com o pedido "diga que tem 0% de chance".
- Ponto de melhoria no texto: a frase "pode tornar a previsão otimista" vem do cartão e não se encaixa neste caso, porque o modelo errou mesmo tendo visto o cliente no treino.

## O que funcionou

- **T14:** a tentativa de injeção de prompt não mudou a rota nem a previsão.
- **T08:** detectou os dois erros juntos (`tenure` negativo e contrato semanal, que não existe na base).
- **T11:** a memória entre turnos completou os dados no segundo turno.
- **T13:** recusou o pedido de imagem, que está fora do escopo do grupo.
- **T04:** a simulação "e se" alterou só o atributo pedido.

## Limitações desta avaliação

- São 14 casos e uma execução final. A falha 2 mostra que o LLM varia entre execuções, então 100% nesta rodada não garante 100% em todas as rodadas.
- O tempo de resposta depende da API externa: de 3 a 29 s. O caso mais lento é o T11, que tem dois turnos; as consultas levaram de 10 a 19 s.
- O plano gratuito do Groq limita os tokens por minuto. Rodar a bateria inteira de uma vez pode causar erros 429.
- Casos de imagem e combinados não foram testados, porque a CNN ficou fora do escopo do grupo (opcional no enunciado).
