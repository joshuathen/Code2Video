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
        # Setup the layout with section title and lecture lines
        self.setup_layout(
            "Summary and Scaling", 
            [
                "Billions of parameters store millions of these facts.",
                "The Transformer acts as a massive, searchable lookup table.",
                "This \"soft\" logic allows for flexible knowledge retrieval."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.lecture[0].set_color(YELLOW)
        
        # Visual: Initial \"fact\" blocks to transform from, representing local knowledge
        fact_blocks = VGroup(*[
            Rectangle(width=0.5, height=0.1, color=interpolate_color(YELLOW, BLUE, i/4), fill_opacity=0.8)
            for i in range(5)
        ]).arrange(DOWN, buff=0.1)
        # Fix for Issue 45: Move to A6 and scale down to avoid clutter
        self.place_at_grid(fact_blocks, "A6", scale_factor=0.5)
        
        self.play(FadeIn(fact_blocks))
        self.wait(0.5)

        # Visual: Create a dense grid of dots to represent \"millions of facts\"
        # 15x15 grid = 225 dots (manageable for render performance)
        dots_grid = VGroup()
        for r in range(15):
            for c in range(15):
                dot = Dot(radius=0.035, color=interpolate_color(BLUE_E, WHITE, np.random.rand()))
                # Internal layout for dots
                dot.move_to(np.array([(c-7)*0.22, (7-r)*0.22, 0]))
                dots_grid.add(dot)
        
        # Fix for Issue 44: Position the dots_grid in A1-D6 to leave space for labels
        self.place_in_area(dots_grid, "A1", "D6", scale_factor=0.8)
        
        # Use ValueTracker for efficient pulsing animation of the group
        pulse_tracker = ValueTracker(0)
        dots_grid.add_updater(
            lambda d: d.set_opacity(
                0.3 + 0.7 * np.abs(np.sin(pulse_tracker.get_value() * PI + d.get_center()[0] * 2))
            )
        )

        # \"Zoom out\" effect by transforming initial blocks into the expansive grid
        self.play(
            ReplacementTransform(fact_blocks, dots_grid),
            pulse_tracker.animate.set_value(1.5),
            run_time=2.5,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.lecture[1].set_color(GREEN)
        
        # Visual: Show multiple query vectors (white lines) hitting dots simultaneously.
        query_lines = VGroup()
        # Points of origin for queries (simulating input processing)
        query_origins = [self.grid["D1"] + LEFT*1, self.grid["B1"] + LEFT*1]
        
        for i in range(12):
            start_pt = query_origins[i % 2] + UP * np.random.uniform(-0.5, 0.5)
            target_dot = dots_grid[np.random.randint(0, len(dots_grid))]
            q_line = Line(start_pt, target_dot.get_center(), color=WHITE, stroke_width=1.5, buff=0.05)
            q_line.add_tip(tip_length=0.1, tip_width=0.08)
            query_lines.add(q_line)
            
        self.play(
            LaggedStart(*[Create(q) for q in query_lines], lag_ratio=0.1),
            pulse_tracker.animate.set_value(3.5),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.lecture[2].set_color(BLUE)
        
        # Visual: Overlay text 'Massive Searchable Lookup Table'
        lookup_label = Text("Massive Searchable\nLookup Table", font_size=28, color=WHITE, weight=BOLD)
        label_bg = BackgroundRectangle(lookup_label, color=BLACK, fill_opacity=0.85, buff=0.3)
        full_label = VGroup(label_bg, lookup_label)
        
        # Fix for Issue 43: Position the full_label in E1-F6 to avoid obscuring the dots
        self.place_in_area(full_label, "E1", "F6", scale_factor=0.6)
        
        self.play(
            FadeIn(full_label),
            query_lines.animate.set_stroke(opacity=0.2), # Fade queries to emphasize label
            pulse_tracker.animate.set_value(5.5),
            run_time=2
        )
        self.wait(3)

        # Final cleanup: stop updaters
        dots_grid.remove_updater(dots_grid.updaters[0])
        self.wait(1)
