from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Initial layout setup for Section 3
        self.setup_layout("Introducing Evidence: The Likelihood Filter", [
            "A new clue appears: a mound of fresh dirt.",
            "We must account for this evidence using a filter.",
            "If the bone is there, a mound is likely.",
            "If it is missing, a mound is still possible.",
            "We draw these outcomes as horizontal slices of probability."
        ])
        
        # Color definitions based on storyboard and cross-section continuity
        color_mound = "#8B4513"       # Saddle Brown for the clue icon
        color_prior_h = "#FFD700"     # Gold for H prior
        color_prior_not_h = "#A9A9A9" # Dark Gray for Not H prior
        color_lime = "#32CD32"        # Lime Green for P(E|H)
        color_orange = "#FF4500"      # Orange Red for P(E|not H)
        color_divider = "#FFFFFF"      # White for highlights
        
        # === Animation for Lecture Line 1 ===
        # Display the 'mound of fresh dirt' icon at the top (A3 grid area)
        mound_icon = VGroup(
            Ellipse(width=0.7, height=0.25, fill_opacity=1, color=color_mound, stroke_width=0),
            Ellipse(width=0.4, height=0.15, fill_opacity=1, color=color_mound, stroke_width=0).shift(UP*0.1)
        )
        # Fix Issue 34: mound_icon scale factor 0.8
        self.place_at_grid(mound_icon, "A3", scale_factor=0.8)
        
        self.play(
            FadeIn(mound_icon),
            self.lecture[0].animate.set_color(color_mound),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Establish the visual anchor system: a square universe split by priors
        square_dim = 3.6
        universe_rect = Rectangle(width=square_dim, height=square_dim, stroke_color=WHITE, stroke_width=1.5)
        # Fix Issue 33 & 35: universe_rect in B2-E4, scale factor 0.9
        self.place_in_area(universe_rect, "B2", "E4", scale_factor=0.9)
        
        # Redefine square_dim after scaling for internal components
        scaled_dim = square_dim * 0.9
        
        # Horizontal bars representing the Priors (Vertical slices)
        # Widths: 20% H, 80% Not H
        slice_h = Rectangle(
            width=0.2 * scaled_dim, 
            height=scaled_dim, 
            fill_opacity=0.1, 
            color=color_prior_h,
            stroke_width=0
        ).align_to(universe_rect, LEFT).align_to(universe_rect, UP)
        
        slice_not_h = Rectangle(
            width=0.8 * scaled_dim, 
            height=scaled_dim, 
            fill_opacity=0.1, 
            color=color_prior_not_h,
            stroke_width=0
        ).align_to(universe_rect, RIGHT).align_to(universe_rect, UP)
        
        label_h = MathTex("P(H)", font_size=18, color=color_prior_h).next_to(slice_h, DOWN, buff=0.1)
        label_not_h = MathTex("P(\\neg H)", font_size=18, color=color_prior_not_h).next_to(slice_not_h, DOWN, buff=0.1)
        
        self.play(
            Create(universe_rect),
            FadeIn(slice_h),
            FadeIn(slice_not_h),
            FadeIn(label_h),
            FadeIn(label_not_h),
            self.lecture[1].animate.set_color(WHITE),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show likelihood for Bone being present: P(E|H) = 0.9 (90% top-filled)
        h_ev_rect = Rectangle(
            width=slice_h.width,
            height=0.9 * slice_h.height,
            fill_opacity=0.8,
            fill_color=color_lime,
            stroke_color=color_lime,
            stroke_width=1
        ).align_to(slice_h, UP).align_to(slice_h, LEFT)
        
        label_ev_h = Text("Mound | Bone", font_size=14, color=color_lime)
        label_ev_h.next_to(h_ev_rect, LEFT, buff=0.15)
        
        self.play(
            FadeIn(h_ev_rect),
            Write(label_ev_h),
            self.lecture[2].animate.set_color(color_lime),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Show likelihood for Bone being elsewhere: P(E|not H) = 0.1 (10% top-filled)
        not_h_ev_rect = Rectangle(
            width=slice_not_h.width,
            height=0.1 * slice_not_h.height,
            fill_opacity=0.8,
            fill_color=color_orange,
            stroke_color=color_orange,
            stroke_width=1
        ).align_to(slice_not_h, UP).align_to(slice_not_h, LEFT)
        
        label_ev_not_h = Text("Mound | No Bone", font_size=14, color=color_orange)
        label_ev_not_h.next_to(not_h_ev_rect, RIGHT, buff=0.15)
        
        self.play(
            FadeIn(not_h_ev_rect),
            Write(label_ev_not_h),
            self.lecture[3].animate.set_color(color_orange),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight horizontal lines and fade labels
        line_h = Line(
            h_ev_rect.get_corner(DL),
            h_ev_rect.get_corner(DR),
            color=color_divider,
            stroke_width=2
        )
        line_not_h = Line(
            not_h_ev_rect.get_corner(DL),
            not_h_ev_rect.get_corner(DR),
            color=color_divider,
            stroke_width=2
        )
        
        self.play(
            Create(line_h),
            Create(line_not_h),
            label_ev_h.animate.set_opacity(0.5),
            label_ev_not_h.animate.set_opacity(0.5),
            self.lecture[4].animate.set_color(color_divider),
            run_time=1.5
        )
        self.wait(2)
