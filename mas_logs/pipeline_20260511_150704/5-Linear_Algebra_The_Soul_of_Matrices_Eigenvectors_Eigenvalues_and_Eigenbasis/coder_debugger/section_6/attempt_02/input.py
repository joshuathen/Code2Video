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
        # Initial layout setup
        lecture_lines = [
            "Eigenvalues represent a system's natural frequencies.",
            "Eigenvectors show the physical shape of vibration.",
            "Avoiding resonance prevents structures from collapsing."
        ]
        self.setup_layout("Real-World Application: Bridge Resonance", lecture_lines)

        # Colors
        COLOR_EIGVAL = "#FF0000"
        COLOR_EIGVEC = "#00FFFF"
        COLOR_RESONANCE = "#FFA500"
        COLOR_BRIDGE = "#FFFFFF"

        # Bridge Geometry parameters
        deck_width = 4.0
        
        # Grid references for area: B2 to E5
        center_pos = self.grid["D4"]
        
        # Value trackers for animation
        time_tracker = ValueTracker(0)
        amp_tracker = ValueTracker(0.0) # Start with no vibration
        freq_tracker = ValueTracker(1.0)

        # Deck: made of multiple segments to allow bending
        num_segments = 20
        deck_segments = VGroup()
        for i in range(num_segments):
            seg = Line(LEFT * 0.5, RIGHT * 0.5, color=COLOR_BRIDGE, stroke_width=4)
            deck_segments.add(seg)
        
        # Function to update deck segment positions
        def update_deck(mob):
            t = time_tracker.get_value()
            A = amp_tracker.get_value()
            f = freq_tracker.get_value()
            
            # Mode shape: sin(pi * x / L)
            L = deck_width
            for i, segment in enumerate(mob):
                x_alpha = i / num_segments
                x_val = (x_alpha - 0.5) * L
                
                # Displacement
                y_disp = A * np.sin(np.pi * x_alpha) * np.sin(2 * np.pi * f * t)
                
                # Calculate start and end for segments
                x_start = x_val + center_pos[0]
                x_end = ( (i+1)/num_segments - 0.5 ) * L + center_pos[0]
                
                y_start = y_disp + center_pos[1]
                # Next point's displacement
                y_disp_next = A * np.sin(np.pi * (i+1)/num_segments) * np.sin(2 * np.pi * f * t)
                y_end = y_disp_next + center_pos[1]
                
                segment.set_points_as_corners([
                    [x_start, y_start, 0],
                    [x_end, y_end, 0]
                ])

        deck_segments.add_updater(update_deck)
        
        # Supports
        support_left = Line(center_pos + LEFT * 2, center_pos + LEFT * 2 + DOWN * 1.5, color=COLOR_BRIDGE)
        support_right = Line(center_pos + RIGHT * 2, center_pos + RIGHT * 2 + DOWN * 1.5, color=COLOR_BRIDGE)
        
        # Labels - Replacing MathTex with Text to avoid LaTeX dependency errors
        lambda_label = Text("λ = Frequency", color=COLOR_EIGVAL, font_size=28)
        v_label = Text("v = Mode Shape", color=COLOR_EIGVEC, font_size=28)
        
        self.place_at_grid(lambda_label, "B5", scale_factor=0.8)
        self.place_at_grid(v_label, "B2", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_EIGVAL)
        self.add(support_left, support_right, deck_segments)
        self.play(Write(lambda_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_EIGVEC)
        self.play(Write(v_label))
        
        # Start small vibration
        self.add_updater(lambda dt: time_tracker.increment_value(dt))
        self.play(amp_tracker.animate.set_value(0.3), run_time=2)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_RESONANCE)
        
        # Resonance visualization (High amplitude)
        res_text = Text("RESONANCE!", color=COLOR_RESONANCE, font_size=36)
        self.place_at_grid(res_text, "D4", scale_factor=1.0)
        
        self.play(
            amp_tracker.animate.set_value(1.2),
            FadeIn(res_text),
            run_time=1.5
        )
        self.play(Indicate(res_text, color=COLOR_RESONANCE))
        self.wait(1)
        
        # Show stabilization (Avoiding resonance)
        safe_text = Text("Safe Frequency", color=WHITE, font_size=24)
        self.place_at_grid(safe_text, "E4", scale_factor=0.8)
        
        self.play(
            amp_tracker.animate.set_value(0.1),
            FadeOut(res_text),
            FadeIn(safe_text),
            run_time=2
        )
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(lambda_label),
            FadeOut(v_label),
            FadeOut(safe_text),
            FadeOut(deck_segments),
            FadeOut(support_left),
            FadeOut(support_right)
        )