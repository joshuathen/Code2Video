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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: Why Do We Need This?", [
            "Smart bases simplify complex problems.",
            "Operations look easier in diagonalization.",
            "Saves significant computation time.",
            "Spaceships need efficient physics calculations.",
            "Optimized basis enables faster physics."
        ])
        
        # Load assets
        spaceship = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spaceship.svg")
        
        # Define objects
        rotation_obj = spaceship.copy()
        calc_text = Text("Calculations", font_size=24, color=YELLOW)
        efficiency_graph = Rectangle(width=2, height=1, color=BLUE, fill_opacity=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF8800"))
        self.place_at_grid(rotation_obj, 'B4', scale_factor=0.9)
        self.play(FadeIn(rotation_obj))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#88FF00"))
        self.place_at_grid(calc_text, 'C4', scale_factor=0.8)
        self.play(Write(calc_text))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#0088FF"))
        self.place_in_area(efficiency_graph, 'D3', 'F6', scale_factor=0.6)
        self.play(Create(efficiency_graph))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        self.play(rotation_obj.animate.rotate(PI/4))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        self.play(rotation_obj.animate.shift(RIGHT * 1))
        self.play(FadeOut(rotation_obj), FadeOut(calc_text), FadeOut(efficiency_graph))
