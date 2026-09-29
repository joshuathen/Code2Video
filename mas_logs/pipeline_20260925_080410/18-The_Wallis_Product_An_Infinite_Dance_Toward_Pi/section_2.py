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
        self.setup_layout("The Recursive Ratio", [
            "I sub n relates to I sub n minus 2.", 
            "The ratio of integrals approaches one.", 
            "This creates our infinite sequence."
        ])
        
        # Assets (Using placeholders as path is non-existent)
        # Note: Since the provided asset path is explicitly '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg',
        # we try to load it. If it fails, we default to a simple shape.
        try:
            asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        except:
            asset_icon = Circle(radius=0.2, color=WHITE)

        # Animations
        # === Animation for Lecture Line 1 ===
        # Show ratio R_n = I_{n+1} / I_n
        ratio_text = MathTex(r"R_n = \frac{I_{n+1}}{I_n}", color=WHITE)
        self.place_in_area(ratio_text, "B2", "B5", scale_factor=1.2)
        icon1 = asset_icon.copy()
        self.place_at_grid(icon1, "B1", scale_factor=0.5)
        self.play(Write(ratio_text), FadeIn(icon1))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Demonstrate R_n approaches 1 as n grows
        arrow = Arrow(start=self.grid["C3"], end=self.grid["C5"], color="#00FF00")
        limit_text = MathTex(r"n \to \infty, R_n \to 1", color="#00FF00")
        self.place_at_grid(limit_text, "C4", scale_factor=1.0)
        icon2 = asset_icon.copy()
        self.place_at_grid(icon2, "C2", scale_factor=0.5)
        self.play(Create(arrow), Write(limit_text), FadeIn(icon2))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the squeeze between (n/(n+1)) and 1
        squeeze = MathTex(r"\frac{n}{n+1} < R_n < 1", color="#FFA500")
        self.place_at_grid(squeeze, "D3", scale_factor=1.0)
        icon3 = asset_icon.copy()
        self.place_at_grid(icon3, "D2", scale_factor=0.5)
        self.play(FadeIn(squeeze), FadeIn(icon3))
        self.play(self.lecture[2].animate.set_color("#FFA500"))
        self.wait(2)
