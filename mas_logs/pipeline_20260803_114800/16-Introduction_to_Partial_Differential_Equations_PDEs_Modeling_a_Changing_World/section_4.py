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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data
        title = "Visualizing the Dynamics: The Vibrating String"
        lecture_lines = [
            "A plucked guitar string demonstrates the wave equation.",
            "Its curvature determines the force pulling it back.",
            "This force creates acceleration, changing its velocity over time.",
            "PDEs link the string's shape to its future motion.",
            "Watch how the wave travels through the material."
        ]
        
        self.setup_layout(title, lecture_lines)

        # Assets & Trackers
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/guitar.svg]
        guitar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/guitar.svg")
        self.place_in_area(guitar, "A2", "E6", scale_factor=1.5)
        guitar.set_opacity(0.15)
        
        time_tracker = ValueTracker(0)
        amp_tracker = ValueTracker(0)
        osc_tracker = ValueTracker(0)
        
        # Colors
        string_color = WHITE
        arrow_color = "#FF0000"
        glow_color = "#ADD8E6"

        # Positioning logic from Issue 28: A2 to E6, scale 0.8
        tl_str = self.grid["A2"]
        br_str = self.grid["E6"]
        center_str = (tl_str + br_str) / 2
        width_str = (br_str[0] - tl_str[0]) * 0.8
        left_x = center_str[0] - width_str / 2
        right_x = center_str[0] + width_str / 2
        center_y = center_str[1]

        # String VMobject
        string = VMobject(color=string_color)
        num_points = 60
        
        def update_string(mob):
            t = time_tracker.get_value()
            a = amp_tracker.get_value()
            o = osc_tracker.get_value()
            current_amp = a * np.cos(6 * t * o)
            
            pts = []
            for i in range(num_points + 1):
                pct = i / num_points
                x = left_x + pct * width_str
                # Standing wave u(x,t) = A * sin(pi * x / L) * cos(wt)
                y = center_y + current_amp * 0.6 * np.sin(np.pi * pct)
                pts.append([x, y, 0])
            mob.set_points_as_corners(pts)

        string.add_updater(update_string)
        update_string(string)

        # Arrows (Force) - Positioned in B2-D5 with scale 0.5 (Issue 29)
        # Reduced density: 3 arrows.
        arrows = VGroup()
        for i in [0.25, 0.5, 0.75]:
            # Scale 0.5 applied to individual arrows or group
            arrow = Arrow(UP, DOWN, color=arrow_color, buff=0, stroke_width=2, max_tip_length_to_length_ratio=0.2)
            arrow.scale(0.5) 
            arrow.pos_pct = i
            arrows.add(arrow)

        def update_arrows(mob_group):
            a = amp_tracker.get_value()
            o = osc_tracker.get_value()
            t = time_tracker.get_value()
            current_amp = a * np.cos(6 * t * o)
            
            for arrow in mob_group:
                x = left_x + arrow.pos_pct * width_str
                y_string = center_y + current_amp * 0.6 * np.sin(np.pi * arrow.pos_pct)
                
                # Length and opacity proportional to displacement
                if abs(current_amp) < 0.05:
                    arrow.set_opacity(0)
                else:
                    arrow.set_opacity(min(1.0, abs(current_amp) * 1.5))
                    arrow.put_start_and_end_on(
                        [x, y_string, 0],
                        [x, center_y, 0]
                    )

        arrows.add_updater(update_arrows)
        
        # Glow
        glow_string = string.copy().set_color(glow_color).set_stroke(width=15, opacity=0)
        self.glow_active = False
        def update_glow(mob):
            mob.set_points(string.get_points())
            if self.glow_active:
                mob.set_opacity(0.3 + 0.2 * np.sin(12 * time_tracker.get_value()))
            else:
                mob.set_opacity(0)
        glow_string.add_updater(update_glow)

        # === Animation for Lecture Line 1 ===
        # "A plucked guitar string demonstrates the wave equation."
        self.play(self.lecture[0].animate.set_color(string_color))
        self.play(FadeIn(guitar), Create(string))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Its curvature determines the force pulling it back."
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(amp_tracker.animate.set_value(1.2), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This force creates acceleration, changing its velocity over time."
        self.play(self.lecture[2].animate.set_color(arrow_color))
        self.play(FadeIn(arrows))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "PDEs link the string's shape to its future motion."
        self.play(self.lecture[3].animate.set_color(WHITE))
        osc_tracker.set_value(1)
        self.add(time_tracker.add_updater(lambda d, dt: d.set_value(d.get_value() + dt)))
        self.wait(4)

        # === Animation for Lecture Line 5 ===
        # "Watch how the wave travels through the material."
        self.play(self.lecture[4].animate.set_color(glow_color))
        self.add(glow_string)
        self.glow_active = True
        self.wait(5)

        # Cleanup
        self.glow_active = False
        time_tracker.clear_updaters()
        self.wait(1)
