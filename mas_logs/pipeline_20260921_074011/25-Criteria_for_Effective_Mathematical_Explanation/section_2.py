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
        self.setup_layout("Criterion 1: The Principle of Gradual Abstraction", 
                          ["Start with concrete physical objects.", 
                           "Transition to geometric models.", 
                           "Generalize with algebraic formulas."])
        
        # === Animation for Lecture Line 1 ===
        # Display text: 'Start with simple concepts' (#FFFFFF) alongside an apple.
        text_1 = Text("Start with simple concepts", font_size=24, color=WHITE)
        self.place_in_area(text_1, 'B4', 'B6', scale_factor=0.8)
        
        apple = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/apple.svg", color=WHITE)
        self.place_at_grid(apple, 'B2', scale_factor=0.5)
        
        self.play(Write(text_1), FadeIn(apple))
        self.play(self.lecture[0].animate.set_color("#FFCC00"))
        
        # === Animation for Lecture Line 2 ===
        # Animate a small dot growing into a complex shape (#00FFFF).
        dot = Dot(color=WHITE)
        self.place_at_grid(dot, 'E4', scale_factor=0.5)
        
        shape = Star(n=5, outer_radius=0.8, inner_radius=0.4, color="#00FFFF")
        self.place_at_grid(shape, 'E5', scale_factor=0.7)
        
        self.play(FadeIn(dot))
        self.play(Transform(dot, shape))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # === Animation for Lecture Line 3 ===
        # Show a 'Gradual' arrow moving from small to big (#FFFFFF) next to a pencil.
        arrow = Arrow(start=self.grid["F2"], end=self.grid["F5"], color=WHITE)
        label = Text("Gradual", font_size=20, color=WHITE).next_to(arrow, UP)
        
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg", color=WHITE)
        self.place_at_grid(pencil, 'F1', scale_factor=0.5)
        
        self.play(GrowArrow(arrow), Write(label), FadeIn(pencil))
        self.play(self.lecture[2].animate.set_color("#FF99FF"))
        self.wait(2)
