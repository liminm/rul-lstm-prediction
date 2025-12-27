

interface SensorDisplayProps {
    value: number;
    label: string;
}


const SensorDisplay = ({ value, label }: SensorDisplayProps) => {
    return (
        <div>
            <span>{label}: </span>
            <span>{value+100}</span>
        </div>
    );
};

export default SensorDisplay;