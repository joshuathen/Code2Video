from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Quantum computers process many states simultaneously.",
            "This enables exponential speed for algorithms.",
            "They explore all paths at once."
        ]
        self.setup_layout("Real-World Application: Quantum Computing", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display a 2-qubit register with logic gates on a [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg]. Color: #00CED1.
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer, 'C4', scale_factor=0.5)
        
        qubits = VGroup(*[Circle(radius=0.2, color="#00CED1", fill_opacity=0.5) for _ in range(4)])
        gate = Rectangle(height=0.4, width=0.8, color="#00CED1", fill_opacity=0.3)
        qubit_gate_group = VGroup(computer, qubits, gate).arrange(DOWN, buff=0.1)
        self.place_in_area(qubit_gate_group, 'B2', 'D5', scale_factor=0.7)
        
        self.play(FadeIn(qubit_gate_group))
        self.play(self.lecture[0].animate.set_color("#00CED1"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show entanglement as lines connecting two distant qubits. Color: #FF6347.
        line1 = Line(qubits[0].get_center(), qubits[3].get_center(), color="#FF6347")
        line2 = Line(qubits[1].get_center(), qubits[2].get_center(), color="#FF6347")
        
        self.play(Create(line1), Create(line2))
        self.play(self.lecture[1].animate.set_color("#FF6347"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate a quantum algorithm result appearing on output qubits using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg]. Color: #ADFF2F.
        result_computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(result_computer, 'F6', scale_factor=0.4)
        
        self.play(FadeIn(result_computer.set_color("#ADFF2F")))
        self.play(self.lecture[2].animate.set_color("#ADFF2F"))
        self.wait(2)
