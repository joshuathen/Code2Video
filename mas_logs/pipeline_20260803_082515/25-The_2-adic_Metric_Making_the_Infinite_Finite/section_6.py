from manim import *
import numpy as np

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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Conclusion: One Math, Many Metrics",
            [
                "Convergence depends on the \"ruler\" used to measure distance.",
                "Different metrics reveal different structures within the same numbers.",
                "These \"p-adic\" systems are vital for modern cryptography."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Show a pair of glasses [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/glasses.svg] in the center.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Resolve Issue 19: Use Asset
        glasses = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glasses.svg")
        glasses.set_color(WHITE)
        
        # Resolve Issue 29: Fix scale_factor to 0.8
        self.place_in_area(glasses, "B2", "D5", scale_factor=0.8)
        
        self.play(DrawBorderThenFill(glasses))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Through the left lens #FF0000, show the sequence 1, 2, 4 diverging.
        # Through the right lens #00FF00, show the sequence 1, 2, 4 converging.
        self.play(self.lecture[1].animate.set_color(BLUE))
        
        # Create visual lens indicators (color overlays)
        # Position them approximately within the SVG lenses
        left_lens_pos = glasses.get_center() + LEFT * 0.9
        right_lens_pos = glasses.get_center() + RIGHT * 0.9
        
        left_lens_color = Circle(radius=0.6, color="#FF0000", fill_opacity=0.3, stroke_width=0)
        left_lens_color.move_to(left_lens_pos)
        
        right_lens_color = Circle(radius=0.6, color="#00FF00", fill_opacity=0.3, stroke_width=0)
        right_lens_color.move_to(right_lens_pos)

        # Diverging sequence (Real numbers perspective)
        div_seq = VGroup(
            Text("1", font_size=20, color="#FF0000"),
            Text("2", font_size=30, color="#FF0000"),
            Text("4", font_size=45, color="#FF0000")
        ).arrange(DOWN, buff=0.15)
        div_seq.move_to(left_lens_pos)
        
        # Converging sequence (2-adic perspective)
        conv_seq = VGroup(
            Text("1", font_size=45, color="#00FF00"),
            Text("2", font_size=30, color="#00FF00"),
            Text("4", font_size=20, color="#00FF00")
        ).arrange(DOWN, buff=0.1)
        conv_seq.move_to(right_lens_pos)

        self.play(
            FadeIn(left_lens_color),
            FadeIn(right_lens_color),
            LaggedStart(*[Write(m) for m in div_seq], lag_ratio=0.5),
            LaggedStart(*[Write(m) for m in conv_seq], lag_ratio=0.5),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display 'One Math, Many Metrics' #FFFFFF in large font.
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        final_text = Text("One Math,\nMany Metrics", font_size=36, color=WHITE)
        # Resolve Issue 30: Fix scale_factor to 0.75
        self.place_in_area(final_text, "E2", "F5", scale_factor=0.75)
        
        self.play(
            FadeOut(glasses),
            FadeOut(left_lens_color),
            FadeOut(right_lens_color),
            FadeOut(div_seq),
            FadeOut(conv_seq),
            Write(final_text)
        )
        self.wait(2)
