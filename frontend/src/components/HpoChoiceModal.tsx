"use client";

interface Props {
  // Expected HPO run time in seconds; undefined while loading, null when unknown.
  estimateSeconds: number | null | undefined;
  onChoose: (useHpo: boolean) => void;
  onClose: () => void;
}

function formatEstimate(seconds: number | null | undefined): string {
  if (seconds === undefined) return "소요 시간 계산 중...";
  if (seconds === null) return "수 분 이상 소요될 수 있음";
  const minutes = Math.max(1, Math.round(seconds / 60));
  return `약 ${minutes}분 소요`;
}

// Shown on EXPLAIN for a user's own image: run with default hyperparameters (fast) or
// tune each explainer's hyperparameters first (HPO, slower but better-scoring maps).
export default function HpoChoiceModal({ estimateSeconds, onChoose, onClose }: Props) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl shadow-2xl w-[480px] max-w-[calc(100vw-2rem)] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
          <h2 className="text-base font-bold text-gray-900">실행 방식 선택</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 transition-colors">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6 flex flex-col gap-3">
          <p className="text-sm text-gray-500 mb-1">업로드한 이미지를 어떤 방식으로 분석할지 선택하세요.</p>

          <button
            onClick={() => onChoose(false)}
            className="text-left rounded-xl border-2 border-gray-200 hover:border-blue-500 hover:bg-blue-50 px-4 py-3 transition-colors"
          >
            <div className="text-sm font-semibold text-gray-900">기본값으로 실행</div>
            <div className="text-xs text-gray-500 mt-0.5">기본 하이퍼파라미터 사용 · 빠름</div>
          </button>

          <button
            onClick={() => onChoose(true)}
            className="text-left rounded-xl border-2 border-gray-200 hover:border-blue-500 hover:bg-blue-50 px-4 py-3 transition-colors"
          >
            <div className="text-sm font-semibold text-gray-900">HPO 적용</div>
            <div className="text-xs text-gray-500 mt-0.5">
              설명 기법별 하이퍼파라미터 최적화 후 실행 · {formatEstimate(estimateSeconds)}
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
