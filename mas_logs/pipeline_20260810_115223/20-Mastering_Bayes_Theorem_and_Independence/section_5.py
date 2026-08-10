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
            "Independence simplifies math; Bayes allows learning from evidence.",
            "Independence is static; Bayes provides a dynamic updating mechanism.",
            "Machine learning classifiers use these foundations for prediction."
        ]
        self.setup_layout("Summary and Conclusion", lecture_lines)
        
        # Load asset
        email_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/email.svg")
        
        # Visuals
        static_bubble = Circle(color="#00FFFF", fill_opacity=0.3).scale(0.5)
        static_label = Text("Static", font_size=18).move_to(static_bubble)
        static_group = VGroup(static_bubble, static_label)
        
        updating_bubble = Circle(color="#00FFFF", fill_opacity=0.3).scale(0.5)
        updating_label = Text("Updating", font_size=18).move_to(updating_bubble)
        updating_group = VGroup(updating_bubble, updating_label)
        
        formula = MathTex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}", color="#FFFF00").scale(0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(FadeIn(static_group), FadeIn(updating_group))
        self.place_at_grid(static_group, "B4")
        self.place_at_grid(updating_group, "B6")
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.place_in_area(formula, "C3", "D5", scale_factor=1.0)
        self.play(Write(formula))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        email_icon.set_color("#FF00FF")
        self.place_at_grid(email_icon, "E4", scale_factor=0.6)
        self.play(FadeIn(email_icon))
        
        self.wait(2)
        
        # Final transition
        final_text = Text("Conclusion", font_size=40, color=WHITE)
        self.play(FadeOut(self.lecture), FadeOut(static_group), FadeOut(updating_group), FadeOut(formula))
        self.play(FadeIn(final_text), email_icon.animate.move_to(ORIGIN).scale(2))
        self.wait(2)
