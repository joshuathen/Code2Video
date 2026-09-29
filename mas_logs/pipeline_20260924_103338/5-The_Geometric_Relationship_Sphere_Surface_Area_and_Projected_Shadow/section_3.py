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

class Section3Scene(TeachingScene):
    def construct(self):
        lines = [
            "Archimedes discovered a remarkable geometric truth.",
            "A sphere's surface is 4 times its shadow.",
            "Four circular shadows cover the sphere exactly.",
            "The total surface area is 4πr².",
            "This elegant ratio defines spherical geometry."
        ]
        self.setup_layout("Deriving the Sphere Surface Area", lines)
        
        # Elements
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=BLUE)
        shadow_circle = Circle(radius=1.2, color=YELLOW, fill_opacity=0.3)
        rects = VGroup(*[Circle(radius=0.5, color="#32CD32", fill_opacity=0.3) for _ in range(4)])
        formula = MathTex(r"A = 4\pi r^2", font_size=42)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(sphere, 'B4', scale_factor=0.5)
        self.play(FadeIn(sphere))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(shadow_circle, 'C4', scale_factor=0.5)
        self.play(FadeIn(shadow_circle))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Using peripheral grid quadrants as per B013
        rects.arrange(RIGHT, buff=0.1)
        self.place_at_grid(rects, 'D4', scale_factor=0.3)
        self.play(FadeIn(rects))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        self.place_at_grid(formula, 'E5', scale_factor=0.8)
        self.play(Write(formula))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        self.play(Indicate(formula))
        self.wait(2)
