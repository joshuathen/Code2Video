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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize layout
        lecture_lines = [
            "Logic, counting, and calculus all point to one answer.",
            "Zero factorial is the glue of mathematical patterns.",
            "Thanks for exploring this mathematical mystery with us!"
        ]
        self.setup_layout("Summary and Conclusion", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 in Cyan
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        # Labels "Patterns", "Counting", and "Consistency"
        patterns_label = Text("Patterns", color="#00FFFF", font_size=32)
        counting_label = Text("Counting", color="#00FFFF", font_size=32)
        consistency_label = Text("Consistency", color="#00FFFF", font_size=32)
        
        # Position sequentially in a vertical list
        self.place_at_grid(patterns_label, "B3", scale_factor=1.0)
        self.place_at_grid(counting_label, "C3", scale_factor=1.0)
        self.place_at_grid(consistency_label, "D3", scale_factor=1.0)
        
        # Appear sequentially
        self.play(FadeIn(patterns_label, shift=UP))
        self.wait(0.5)
        self.play(FadeIn(counting_label, shift=UP))
        self.wait(0.5)
        self.play(FadeIn(consistency_label, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight to line 2 (Gold)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700")
        )
        
        # Remove previous list items
        self.play(
            FadeOut(patterns_label),
            FadeOut(counting_label),
            FadeOut(consistency_label)
        )
        
        # Display final equation 0! = 1 with Success asset
        # Replacement for missing Success_1.png. Used Text instead of MathTex to avoid 'latex' dependency.
        success_icon = Text("✓", color=GREEN, font_size=60)
        self.place_in_area(success_icon, "B2", "E5", scale_factor=1.5)
        
        # Equation in large, bold gold font. Used Text instead of MathTex to avoid 'latex' dependency.
        final_eq = Text("0! = 1", color="#FFD700", font_size=90, weight=BOLD)
        self.place_in_area(final_eq, "B2", "E5")
        
        self.play(
            FadeIn(success_icon),
            Write(final_eq)
        )
        
        # Subtle pulse effect (scaling up and down)
        self.play(final_eq.animate.scale(1.15), run_time=0.4)
        self.play(final_eq.animate.scale(1/1.15), run_time=0.4)
        self.play(final_eq.animate.scale(1.15), run_time=0.4)
        self.play(final_eq.animate.scale(1/1.15), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture highlight to line 3 (White)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        
        # Fade screen to black (clear scene elements)
        self.play(
            FadeOut(self.lecture),
            FadeOut(self.title),
            FadeOut(success_icon),
            FadeOut(final_eq),
            run_time=1.5
        )
        
        # Display "Thank You!" at the center of the animation area
        thanks_text = Text("Thank You!", color="#FFFFFF", font_size=48)
        self.place_in_area(thanks_text, "A1", "F6")
        
        self.play(Write(thanks_text))
        self.wait(3)
