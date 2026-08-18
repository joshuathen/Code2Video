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
        lines = ["The forward process adds Gaussian noise.", "Pure static replaces the original image.", "The reverse process iteratively removes noise.", "Markov chains guide the image reconstruction.", "Finally, a clear image emerges."]
        self.setup_layout("The Mathematics of Diffusion", lines)
        
        # Visual assets (SVGMobject used for .svg files)
        robot_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/image.svg")
        noise_rect = Rectangle(width=2, height=2, fill_opacity=0.5, color=GRAY)
        formula = MathTex(r"x_t = \sqrt{1-\beta}x_0 + \sqrt{\beta}\epsilon", font_size=32, color=GREEN)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(robot_img, 'B2', scale_factor=0.7)
        self.play(FadeIn(robot_img))
        self.play(self.lecture[0].animate.set_color("#808080"))
        
        # === Animation for Lecture Line 2 ===
        self.place_at_grid(noise_rect, 'C4', scale_factor=0.8)
        self.play(FadeIn(noise_rect))
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(noise_rect), FadeOut(robot_img))
        self.place_in_area(formula, 'D2', 'E5', scale_factor=0.9)
        self.play(Write(formula))
        self.play(self.lecture[2].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        final_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/image.svg")
        self.place_at_grid(final_img, 'B2', scale_factor=0.7)
        self.play(FadeIn(final_img))
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        self.wait(1)
