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
        # Setup layout with specific title and lines for section 3
        self.setup_layout("The Twist: Breaking the Pattern", [
            "Surely, six points will result in thirty-two regions?",
            "Let's add the sixth point and count carefully.",
            "We count the outer regions and then the inner ones.",
            "Surprisingly, the total count is only thirty-one regions!",
            "Our beautiful doubling pattern has suddenly failed us."
        ])

        # Colors
        COLOR_CIRCLE = "#FFFFFF"
        COLOR_CHORDS = "#FFFF00"
        COLOR_POINTER = "#FF0000"
        COLOR_PULSE = "#00FF00"
        COLOR_TEXT_REVEAL = "#FF0000"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        circle_radius = 2.3
        self.circle = Circle(radius=circle_radius, color=COLOR_CIRCLE)
        # Fix for Issue 26: Use 'A1' to 'E6' and scale_factor 0.9 to avoid overlaps
        self.place_in_area(self.circle, 'A1', 'E6', scale_factor=0.9)
        center = self.circle.get_center()

        # Non-uniform angles to avoid triple intersections (B059)
        angles = [0, 45, 105, 160, 220, 295]
        # Points scale with the circle (0.9 scale factor)
        eff_radius = circle_radius * 0.9
        points = [center + eff_radius * np.array([np.cos(a*DEGREES), np.sin(a*DEGREES), 0]) for a in angles]
        dots = VGroup(*[Dot(p, color=COLOR_CIRCLE, radius=0.08) for p in points])
        
        chords = VGroup()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                chords.add(Line(points[i], points[j], color=COLOR_CHORDS, stroke_width=2))

        self.play(Create(self.circle), run_time=1)
        self.play(Create(dots), run_time=1)
        self.play(Create(chords), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Define 31 approximate counting locations within the circle
        outer_locs = [center + (eff_radius * 0.85) * np.array([np.cos((a+22)*DEGREES), np.sin((a+22)*DEGREES), 0]) for a in angles]
        mid_locs = [center + (eff_radius * 0.5) * np.array([np.cos((a+12)*DEGREES), np.sin((a+12)*DEGREES), 0]) for a in np.linspace(0, 360, 10, endpoint=False)]
        inner_locs = [center + (eff_radius * 0.2) * np.array([np.cos((a+5)*DEGREES), np.sin((a+5)*DEGREES), 0]) for a in np.linspace(0, 360, 15, endpoint=False)]
        
        all_locs = outer_locs + mid_locs + inner_locs
        all_locs = all_locs[:31]

        # Fix for Issue 19: Use asset pointer.svg
        self.pointer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pointer.svg")
        self.pointer.set_color(COLOR_POINTER).scale(0.3).set_z_index(10)
        self.pointer.move_to(all_locs[0])
        
        self.counter_val = Integer(1).next_to(self.pointer, UP, buff=0.1).set_color(COLOR_PULSE).scale(0.8)
        
        self.add(self.pointer, self.counter_val)

        def pulse_at(pos):
            p = Circle(radius=0.05, color=COLOR_PULSE, stroke_width=4).move_to(pos)
            self.add(p)
            self.play(p.animate.scale(8).set_stroke(opacity=0), run_time=0.3)
            self.remove(p)

        # Count 1-15
        for i in range(15):
            target_pos = all_locs[i]
            if i == 0:
                pulse_at(target_pos)
            else:
                self.play(
                    self.pointer.animate.move_to(target_pos),
                    self.counter_val.animate.set_value(i+1).next_to(target_pos, UP, buff=0.1),
                    run_time=0.2
                )
                pulse_at(target_pos)
        
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        for i in range(15, 30):
            target_pos = all_locs[i]
            self.play(
                self.pointer.animate.move_to(target_pos),
                self.counter_val.animate.set_value(i+1).next_to(target_pos, UP, buff=0.1),
                run_time=0.2
            )
            pulse_at(target_pos)
        
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        target_pos = all_locs[30]
        self.play(
            self.pointer.animate.move_to(target_pos),
            self.counter_val.animate.set_value(31).next_to(target_pos, UP, buff=0.1),
            run_time=0.4
        )
        p_final = Circle(radius=0.05, color=COLOR_PULSE, stroke_width=6).move_to(target_pos)
        self.add(p_final)
        self.play(p_final.animate.scale(12).set_stroke(opacity=0), run_time=0.6)
        self.remove(p_final)
        
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Fix for Issue 27: result_text at 'F2', scale 0.8
        self.result_text = MathTex("R = 31", color=COLOR_TEXT_REVEAL)
        self.place_at_grid(self.result_text, 'F2', scale_factor=0.8) 
        
        # Fix for Issue 28: break_pattern at 'F4', scale 0.8
        self.break_pattern = MathTex("31 \\neq 2^{6-1}", color=COLOR_TEXT_REVEAL)
        self.place_at_grid(self.break_pattern, 'F4', scale_factor=0.8)

        self.play(Write(self.result_text))
        self.play(Write(self.break_pattern))
        self.wait(2)

        # Final state color preservation
        self.wait(1)
