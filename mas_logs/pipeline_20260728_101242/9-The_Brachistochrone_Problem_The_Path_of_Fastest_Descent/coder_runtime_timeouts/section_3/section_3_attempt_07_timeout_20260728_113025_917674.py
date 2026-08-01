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
        # Setup the layout with the specific lecture lines for Section 3
        title = "The Physics: Energy to Velocity"
        lines = [
            "In gravity, potential energy converts to kinetic energy.",
            "Velocity depends on the vertical distance fallen.",
            "This links depth directly to the speed of travel."
        ]
        self.setup_layout(title, lines)
        
        # Grid positions for the wire
        start_pt = self.grid["B2"]
        end_pt = self.grid["E5"]
        # Use a Bezier curve for the wire - faster than FunctionGraph
        # Control point creates a nice downward curve
        control_pt = np.array([start_pt[0] + 2, start_pt[1], 0])
        wire_path = CubicBezier(start_pt, control_pt, control_pt, end_pt, color="#888888")
        
        # Bead
        bead = Dot(color="#FFFFFF", radius=0.1)
        bead.move_to(start_pt)
        
        # Trackers
        # Using alpha (0 to 1) for movement along the curve
        alpha_tracker = ValueTracker(0)
        
        # Vertical drop indicator (y)
        y_bar = Line(start_pt, start_pt, color="#FF00FF", stroke_width=4)
        y_label = Text("y", color="#FF00FF", font_size=22, slant=ITALIC)
        y_label.move_to(start_pt + LEFT * 0.3)
        
        # Physics formula
        formula = MathTex(r"v = \sqrt{2gy}", color="#00FFFF", font_size=32)
        self.place_at_grid(formula, "A5", scale_factor=1.0)
        
        # Velocity arrow
        # Pre-create the arrow and update it in place
        v_arrow = Arrow(start_pt, start_pt + RIGHT, buff=0, color="#00FFFF", stroke_width=3, max_tip_length_to_length_ratio=0.2)
        v_arrow.set_opacity(0)

        # Energy Labels
        pe_label = Text("PE", font_size=20, color=WHITE)
        self.place_at_grid(pe_label, "A2", scale_factor=0.8)
        ke_label = Text("KE", font_size=20, color=WHITE)

        # Optimized Updaters
        def update_bead(m):
            a = alpha_tracker.get_value()
            m.move_to(wire_path.point_from_proportion(a))
            
        def update_y_bar(m):
            a = alpha_tracker.get_value()
            curr_pos = wire_path.point_from_proportion(a)
            # Bar from the top height to the current bead height
            m.put_start_and_end_on(
                [curr_pos[0], start_pt[1], 0],
                [curr_pos[0], curr_pos[1], 0]
            )
            
        def update_y_label(m):
            a = alpha_tracker.get_value()
            curr_pos = wire_path.point_from_proportion(a)
            m.move_to([curr_pos[0] - 0.3, (start_pt[1] + curr_pos[1])/2, 0])
            
        def update_v_arrow(m):
            a = alpha_tracker.get_value()
            if a < 0.01:
                m.set_opacity(0)
                return
            m.set_opacity(1)
            
            p0 = wire_path.point_from_proportion(max(0, a - 0.01))
            p1 = wire_path.point_from_proportion(min(1, a + 0.01))
            tangent = p1 - p0
            norm = np.linalg.norm(tangent)
            if norm > 0:
                unit_tangent = tangent / norm
                # y_drop is height difference
                y_drop = max(0, start_pt[1] - p1[1])
                # Speed is proportional to sqrt(y_drop)
                speed = np.sqrt(y_drop) * 0.8
                m.put_start_and_end_on(p1, p1 + unit_tangent * speed)
            
        def update_ke_label(m):
            a = alpha_tracker.get_value()
            curr_pos = wire_path.point_from_proportion(a)
            m.move_to(curr_pos + DR * 0.3)

        # === Animation for Lecture Line 1 ===
        # "In gravity, potential energy converts to kinetic energy."
        self.lecture[0].set_color(WHITE)
        self.play(
            Create(wire_path),
            FadeIn(bead),
            FadeIn(pe_label),
            run_time=0.8
        )
        self.wait(0.2)

        # === Animation for Lecture Line 2 ===
        # "Velocity depends on the vertical distance fallen."
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FF00FF") 
        
        y_bar.add_updater(update_y_bar)
        y_label.add_updater(update_y_label)
        
        self.play(
            FadeIn(y_bar),
            FadeIn(y_label),
            run_time=0.8
        )
        self.wait(0.2)

        # === Animation for Lecture Line 3 ===
        # "This links depth directly to the speed of travel."
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#00FFFF") 
        
        bead.add_updater(update_bead)
        v_arrow.add_updater(update_v_arrow)
        ke_label.add_updater(update_ke_label)
        
        self.play(
            FadeIn(formula),
            FadeIn(v_arrow),
            FadeIn(ke_label),
            run_time=0.8
        )
        
        # Descent animation - move alpha from 0 to 1
        self.play(
            alpha_tracker.animate.set_value(1),
            run_time=2.5,
            rate_func=slow_into
        )
        
        # Clean up updaters for stable end state
        bead.clear_updaters()
        y_bar.clear_updaters()
        v_arrow.clear_updaters()
        y_label.clear_updaters()
        ke_label.clear_updaters()
        
        self.wait(1.5)
