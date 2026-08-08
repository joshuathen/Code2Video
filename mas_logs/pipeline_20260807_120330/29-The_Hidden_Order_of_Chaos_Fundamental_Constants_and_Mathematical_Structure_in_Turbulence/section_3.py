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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "The Energy Cascade: Richardson’s Russian Dolls"
        lines = [
            "Energy enters flow through large-scale eddies.",
            "These large swirls fracture into smaller ones.",
            "This process is called the energy cascade.",
            "Big whorls have little whorls feeding on velocity.",
            "Smaller whorls continue the chain down the scale."
        ]
        self.setup_layout(title, lines)

        # Highlighting colors
        HIGHLIGHT = YELLOW
        NORMAL = WHITE

        # === Animation for Lecture Line 1 ===
        # Energy enters flow through large-scale eddies.
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT))
        
        large_eddy = Circle(radius=1.5, color="#0000FF", stroke_width=4)
        # Add an arrow inside to show rotation
        arrow = Arc(radius=1.2, start_angle=0, angle=TAU*0.8, color="#0000FF").add_tip()
        large_vortex = VGroup(large_eddy, arrow)
        
        # Fix Issue 30: Use place_in_area for better centering and size
        self.place_in_area(large_vortex, 'B2', 'E5', scale_factor=1.2)
        
        # Continuous rotation using ValueTracker
        rot_tracker = ValueTracker(1)
        large_vortex.add_updater(lambda m, dt: m.rotate(rot_tracker.get_value() * dt))
        
        self.play(Create(large_vortex))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # These large swirls fracture into smaller ones.
        self.play(
            self.lecture[0].animate.set_color(NORMAL),
            self.lecture[1].animate.set_color(HIGHLIGHT)
        )
        
        # Split into two medium-sized yellow rotating circles
        medium_vortices = VGroup()
        for _ in range(2):
            y_circle = Circle(radius=0.7, color="#FFFF00", stroke_width=3)
            y_arrow = Arc(radius=0.5, start_angle=0, angle=TAU*0.8, color="#FFFF00").add_tip()
            y_vortex = VGroup(y_circle, y_arrow)
            medium_vortices.add(y_vortex)
        
        medium_vortices.arrange(RIGHT, buff=0.5)
        # Fix Issue 31: use place_in_area to avoid tight packing
        self.place_in_area(medium_vortices, 'B2', 'E5', scale_factor=0.9)
        
        medium_rot_tracker = ValueTracker(2.5)
        for v in medium_vortices:
            v.add_updater(lambda m, dt: m.rotate(medium_rot_tracker.get_value() * dt))

        self.play(
            FadeOut(large_vortex),
            FadeIn(medium_vortices),
            run_time=1.2
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # This process is called the energy cascade.
        self.play(
            self.lecture[1].animate.set_color(NORMAL),
            self.lecture[2].animate.set_color(HIGHLIGHT)
        )
        
        # Visualizing fragmentation with white arrows
        cascade_arrows = VGroup()
        for v in medium_vortices:
            # Arrows pointing outwards to suggest breakdown
            arr1 = Arrow(v.get_center(), v.get_center() + DR*0.7, color=WHITE, stroke_width=2, buff=0.1)
            arr2 = Arrow(v.get_center(), v.get_center() + DL*0.7, color=WHITE, stroke_width=2, buff=0.1)
            cascade_arrows.add(arr1, arr2)
        
        self.play(Create(cascade_arrows))
        self.wait(1.5)
        self.play(FadeOut(cascade_arrows))

        # === Animation for Lecture Line 4 ===
        # Big whorls have little whorls feeding on velocity.
        self.play(
            self.lecture[2].animate.set_color(NORMAL),
            self.lecture[3].animate.set_color(HIGHLIGHT)
        )
        
        # Replace medium vortices with 16 small green dots
        # Fix Issue 32: small_vortices (16 green dots) using place_in_area
        small_vortices = VGroup()
        for _ in range(16):
            g_dot = Circle(radius=0.18, color="#00FF00", stroke_width=2, fill_opacity=0.3)
            small_vortices.add(g_dot)
        
        small_vortices.arrange_in_grid(rows=4, cols=4, buff=0.2)
        self.place_in_area(small_vortices, 'A2', 'F5', scale_factor=1.0)
        
        small_rot_tracker = ValueTracker(5)
        for g in small_vortices:
            g.add_updater(lambda m, dt: m.rotate(small_rot_tracker.get_value() * dt))
        
        self.play(
            FadeOut(medium_vortices),
            LaggedStart(*[FadeIn(g) for g in small_vortices], lag_ratio=0.05),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Smaller whorls continue the chain down the scale.
        self.play(
            self.lecture[3].animate.set_color(NORMAL),
            self.lecture[4].animate.set_color(HIGHLIGHT)
        )
        
        # Fill the remaining visual area with tiny spinning white dots
        tiny_dots = VGroup()
        for _ in range(60):
            dot = Dot(radius=0.02, color=WHITE)
            tiny_dots.add(dot)
            
        tiny_dots.arrange_in_grid(rows=10, cols=6, buff=0.25)
        # Using A2 to F6 to respect B021 and fill right space
        self.place_in_area(tiny_dots, 'A2', 'F6', scale_factor=1.1)
        
        self.play(
            small_rot_tracker.animate.set_value(12),
            FadeIn(tiny_dots, lag_ratio=0.01),
            run_time=2
        )
        self.wait(2)
        
        # Cleanup for section end
        self.play(
            FadeOut(small_vortices),
            FadeOut(tiny_dots),
            self.lecture[4].animate.set_color(NORMAL)
        )
        self.wait(1)
