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
        # Setup layout
        title_str = "Conclusion: The Beauty of Combinatorial Geometry"
        lines = [
            "Simple invariants can explain complex geometric systems.",
            "This is the famous 2011 IMO Windmill Problem.",
            "Mathematical beauty emerges from these unchanging patterns."
        ]
        self.setup_layout(title_str, lines)

        # Colors for highlights
        HIGHLIGHT_COLOR = "#FFFF00"
        STAR_COLOR = "#FFEEBB"
        WEB_COLOR = "#44AAFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))

        # Generate a 'web' of lines
        # First, define points (stars)
        point_coords = [
            'B2', 'B5', 'C3', 'D1', 'D6', 'E3', 'E5', 'F2'
        ]
        stars = VGroup()
        for pos in point_coords:
            star = Dot(point=self.grid[pos], color=STAR_COLOR, radius=0.06)
            # Add a slight glow effect
            glow = Dot(point=self.grid[pos], color=STAR_COLOR, radius=0.12, fill_opacity=0.3)
            stars.add(VGroup(glow, star))

        # Define connections for the 'web'
        connections = [
            ('B2', 'D6'), ('D6', 'F2'), ('F2', 'B5'), 
            ('B5', 'E3'), ('E3', 'D1'), ('D1', 'E5'),
            ('E5', 'C3'), ('C3', 'B2')
        ]
        
        web_lines = VGroup()
        for start_key, end_key in connections:
            line = Line(self.grid[start_key], self.grid[end_key], color=WEB_COLOR, stroke_width=2, stroke_opacity=0.6)
            web_lines.add(line)

        # Show stars and draw the web
        self.play(FadeIn(stars, lag_ratio=0.1))
        self.play(Create(web_lines, lag_ratio=0.2, run_time=3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )

        # Overlay text - Resolved Issues 41 and 42 by moving positions
        invariants_text = Text("The Power of Invariants", font_size=32, color=WHITE)
        imo_text = Text("IMO 2011", font_size=28, color=WHITE)
        
        self.place_in_area(invariants_text, "A2", "A5", scale_factor=0.7)
        self.place_in_area(imo_text, "B2", "B5", scale_factor=0.7)

        # Add a subtle background box - Resolved Issue 40 by moving to A2-B5
        # Adjusted height/width to fit the new area precisely
        text_bg = RoundedRectangle(corner_radius=0.2, height=1.6, width=4.0, fill_color=BLACK, fill_opacity=0.7, stroke_width=1)
        self.place_in_area(text_bg, "A2", "B5")

        self.play(FadeIn(text_bg), Write(invariants_text), Write(imo_text))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )

        # Fade scene into a star field
        # Create many more tiny stars
        star_field = VGroup()
        for _ in range(40):
            # Random position within the grid area
            rand_x = np.random.uniform(1.0, 6.0)
            rand_y = np.random.uniform(-3.0, 2.5)
            small_star = Dot(point=[rand_x, rand_y, 0], radius=np.random.uniform(0.01, 0.04), color=WHITE, fill_opacity=np.random.uniform(0.2, 0.8))
            star_field.add(small_star)

        self.play(
            FadeOut(web_lines),
            FadeOut(text_bg),
            FadeOut(invariants_text),
            FadeOut(imo_text),
            FadeIn(star_field, lag_ratio=0.05),
            run_time=2
        )
        
        # Final slow pulse of the star field
        self.play(
            star_field.animate.set_opacity(0.4),
            stars.animate.scale(1.1).set_opacity(0.8),
            run_time=3,
            rate_func=there_and_back
        )
        self.wait(2)
