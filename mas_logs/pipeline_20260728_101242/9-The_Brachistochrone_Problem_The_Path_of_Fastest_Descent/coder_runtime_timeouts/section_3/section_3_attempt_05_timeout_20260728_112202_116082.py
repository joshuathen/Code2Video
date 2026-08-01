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
        
        # Define constants for the wire curve using the grid system
        x_start = self.grid["B2"][0]
        y_start = self.grid["B2"][1]
        x_end = self.grid["E5"][0]
        y_end = self.grid["E5"][1]
        
        # y = y_start - k_const * (x - x_start)^2
        k_const = (y_start - y_end) / (x_end - x_start)**2
        
        def get_wire_y(x_val):
            return y_start - k_const * (x_val - x_start)**2
            
        wire_path = ParametricFunction(
            lambda t: np.array([t, get_wire_y(t), 0]),
            t_range=[x_start, x_end],
            color="#888888"
        )
        
        # Bead and Labels
        bead = Dot(color="#FFFFFF", radius=0.1)
        bead.move_to(self.grid["B2"])
        
        pe_label = Text("PE", font_size=20, color=WHITE)
        self.place_at_grid(pe_label, "A2", scale_factor=0.8)
        
        ke_label = Text("KE", font_size=20, color=WHITE)
        
        # Trackers
        x_tracker = ValueTracker(x_start)
        
        # Vertical drop indicator
        y_bar = Line(color="#FF00FF", stroke_width=4)
        y_label = Text("y", color="#FF00FF", font_size=22, slant=ITALIC)
        
        # Physics formula
        formula = MathTex(r"v = \sqrt{2gy}", color="#00FFFF", font_size=32)
        self.place_at_grid(formula, "A5", scale_factor=1.0)
        
        # Velocity vector
        v_arrow = Arrow(buff=0, color="#00FFFF", stroke_width=3, max_tip_length_to_length_ratio=0.2)

        # Updaters (Optimized to avoid unnecessary object creation)
        def update_bead(m):
            xv = x_tracker.get_value()
            m.move_to([xv, get_wire_y(xv), 0])
            
        def update_y_bar(m):
            xv = x_tracker.get_value()
            y_curr = get_wire_y(xv)
            m.put_start_and_end_on([xv, y_start, 0], [xv, y_curr, 0])
            
        def update_y_label(m):
            xv = x_tracker.get_value()
            y_curr = get_wire_y(xv)
            m.move_to([xv - 0.4, (y_start + y_curr)/2, 0])
            
        def update_v_arrow(m):
            xv = x_tracker.get_value()
            if xv <= x_start + 0.01:
                m.set_opacity(0)
                return
            m.set_opacity(1)
            yv = get_wire_y(xv)
            # dy/dx = -2 * k_const * (xv - x_start)
            slope = -2.0 * k_const * (xv - x_start)
            norm = np.sqrt(1 + slope**2)
            tx = 1.0/norm
            ty = slope/norm
            # Speed prop to sqrt(y_drop) = sqrt(k_const*(xv-x_start)^2) = (xv-x_start)*const
            speed = (xv - x_start) * 0.6 
            m.put_start_and_end_on([xv, yv, 0], [xv + tx*speed, yv + ty*speed, 0])
            
        def update_ke_label(m):
            xv = x_tracker.get_value()
            yv = get_wire_y(xv)
            m.move_to([xv + 0.4, yv - 0.3, 0])

        # === Animation for Lecture Line 1 ===
        # "In gravity, potential energy converts to kinetic energy."
        self.lecture[0].set_color(WHITE)
        self.play(
            Create(wire_path),
            FadeIn(bead),
            FadeIn(pe_label),
            run_time=0.8
        )
        self.wait(0.4)

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
        self.wait(0.4)

        # === Animation for Lecture Line 3 ===
        # "This links depth directly to the speed of travel."
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#00FFFF") 
        
        bead.add_updater(update_bead)
        v_arrow.add_updater(update_v_arrow)
        ke_label.add_updater(update_ke_label)
        
        self.play(
            Write(formula),
            FadeIn(v_arrow),
            FadeIn(ke_label),
            run_time=0.8
        )
        
        # Descent animation
        self.play(
            x_tracker.animate.set_value(x_end),
            run_time=1.8,
            rate_func=linear
        )
        
        self.wait(1.2)
