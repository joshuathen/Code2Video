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
        self.setup_layout("Real-World Application: Quantum Computing", [
            "Superposition enables parallel computation.",
            "Qubits explore multiple paths simultaneously.",
            "This drastically speeds up complex calculations."
        ])
        
        # --- Mobjects ---
        # Circuit for Lecture 1 (Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg)
        # We load the SVG and place it.
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        circuit = VGroup(
            computer_icon,
            Line(LEFT*0.5, RIGHT*0.5),
            Square(side_length=0.2, fill_opacity=0.5, color=BLUE).shift(UP*0.3),
            Dot(LEFT*0.5), Dot(RIGHT*0.5)
        )
        self.place_at_grid(circuit, 'B4', scale_factor=0.7)
        
        # Qubit/Paths for Lecture 2
        qubit = Dot(color=YELLOW)
        paths = VGroup(*[Line(ORIGIN, UP*0.5+RIGHT*i*0.5) for i in [-1, 0, 1]])
        self.place_at_grid(paths, 'C3', scale_factor=0.8)
        self.place_at_grid(qubit, 'C3', scale_factor=0.8)
        
        # Graph for Lecture 3
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 1, 0.5], x_length=2, y_length=1.5).add_coordinates()
        graph = axes.plot(lambda x: 0.5 * np.exp(-(x-2)**2), color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(circuit))
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(paths), FadeIn(qubit))
        self.play(qubit.animate.move_to(paths[1].get_end()))
        self.play(self.lecture[1].animate.set_color(YELLOW))

        # === Animation for Lecture Line 3 ===
        self.place_in_area(axes, 'D2', 'F3', scale_factor=0.5)
        self.place_in_area(graph, 'D4', 'F5', scale_factor=0.5)
        self.play(Create(axes), Create(graph))
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        self.wait(2)
