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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Vector spaces follow eight formal axioms.", "Closure ensures results stay within the space.", "Associativity and commutativity simplify calculations."]
        self.setup_layout("The Eight Axioms: Defining the Space", lecture_lines)
        
        # Load Assets
        notebook = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/notebook.svg")
        container = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/container.svg")
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        # Show axioms with notebook icon
        self.place_at_grid(notebook, "A1", scale_factor=0.5)
        self.play(FadeIn(notebook))
        axioms = VGroup(*[Text(f"Axiom {i+1}", font_size=18, color=WHITE) for i in range(8)])
        positions = ["B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4"]
        axiom_group = VGroup()
        for i, ax in enumerate(axioms):
            self.place_at_grid(ax, positions[i])
            axiom_group.add(ax)
        self.play(Write(axiom_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF6347"))
        self.place_at_grid(container, "E1", scale_factor=0.6)
        self.play(FadeIn(container), Indicate(axiom_group[0], color="#FF6347"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#87CEEB"))
        self.place_at_grid(calculator, "E3", scale_factor=0.6)
        # Display A+B = B+A
        comm_text = Text("A+B = B+A", font_size=20, color="#87CEEB")
        self.place_at_grid(comm_text, "E4")
        self.play(FadeIn(calculator), Write(comm_text))
        
        self.wait(2)
