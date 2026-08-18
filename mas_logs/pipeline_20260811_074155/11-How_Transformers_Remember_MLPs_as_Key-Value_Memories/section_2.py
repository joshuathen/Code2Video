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
        self.setup_layout(
            "Prerequisite: The Dot Product",
            [
                "The dot product measures similarity between two vectors.",
                "Aligned vectors produce a high positive score.",
                "High scores signal that the model recognizes a concept."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Define vectors
        # Fixed vector (France)
        france_vec = Arrow(start=LEFT*1.5, end=RIGHT*1.5, color=WHITE, stroke_width=6, buff=0)
        self.place_in_area(france_vec, 'B2', 'B5')
        
        # Rotating vector (Pattern)
        pattern_vec = Arrow(start=LEFT*1.5, end=RIGHT*1.5, color=WHITE, stroke_width=6, buff=0)
        self.place_in_area(pattern_vec, 'C2', 'C5')
        
        # Initial state: pattern_vec at an angle
        initial_angle = PI/3
        pattern_vec.rotate(initial_angle, about_point=pattern_vec.get_center())

        self.play(Create(france_vec), Create(pattern_vec))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Similarity Bar setup (Issue 33: Move to Row E to better utilize space)
        bar_bg = Rectangle(width=3, height=0.4, color=WHITE, stroke_width=2)
        self.place_in_area(bar_bg, 'E2', 'E5')
        
        # Similarity Value Tracker (Cosine of angle difference)
        # Using cosine similarity as a proxy for the dot product visual
        similarity_tracker = ValueTracker(np.cos(initial_angle))
        
        # Bar Fill
        bar_fill = Rectangle(
            width=3,
            height=0.4,
            fill_color="#00FF00",
            fill_opacity=0.8,
            stroke_width=0
        )
        # Updater to keep the bar filled according to the similarity tracker
        bar_fill.add_updater(lambda m: m.stretch_to_fit_width(
            max(0.01, bar_bg.width * similarity_tracker.get_value())
        ).align_to(bar_bg, LEFT))
        
        bar_label = Text("Similarity", font_size=18).next_to(bar_bg, UP, buff=0.1)

        self.play(Create(bar_bg), FadeIn(bar_label))
        self.add(bar_fill)
        
        # Rotate vectors to align (make parallel)
        # Issue 32 fix: Vectors are in separate rows (B and C) to avoid overlap when aligned
        self.play(
            Rotate(pattern_vec, -initial_angle, about_point=pattern_vec.get_center()),
            similarity_tracker.animate.set_value(1.0),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Labels (Issue 31: Positioned at B6 and C6 to avoid overlap and clarify which vector is which)
        france_label = Text("France", font_size=24, color=WHITE)
        self.place_at_grid(france_label, 'B6', scale_factor=0.6)
        
        pattern_label = Text("Pattern", font_size=24, color=WHITE)
        self.place_at_grid(pattern_label, 'C6', scale_factor=0.6)

        self.play(Write(france_label), Write(pattern_label))
        
        # Pulse the Similarity bar to emphasize recognition
        self.play(
            bar_fill.animate.scale(1.1),
            bar_bg.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=1
        )
        self.wait(2)

        # Cleanup
        self.play(FadeOut(VGroup(france_vec, pattern_vec, bar_bg, bar_fill, bar_label, france_label, pattern_label)))
