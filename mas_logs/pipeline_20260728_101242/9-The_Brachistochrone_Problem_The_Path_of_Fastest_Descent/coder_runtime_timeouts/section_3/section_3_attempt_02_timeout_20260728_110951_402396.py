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
        self.setup_layout("The Physics: Energy to Velocity", [
            "In gravity, potential energy converts to kinetic energy.",
            "Velocity depends on the vertical distance fallen.",
            "This links depth directly to the speed of travel."
        ])
        
        # Wire: y = -1/3 * (x - 1.5)**2 + 1.2
        # This curve goes from (1.5, 1.2) to (4.5, -1.8)
        def wire_func(x):
            return -0.333 * (x - 1.5)**2 + 1.2
            
        wire_path = FunctionGraph(wire_func, x_range=[1.5, 4.5], color="#888888")
        
        bead = Dot(color="#FFFFFF", radius=0.1)
        bead.move_to(np.array([1.5, 1.2, 0]))
        
        pe_label = Text("PE", font_size=20, color=WHITE)
        self.place_at_grid(pe_label, "B1", scale_factor=1.0)
        pe_label.shift(DOWN * 0.3)
        
        ke_label = Text("KE", font_size=20, color=WHITE)
        
        # Pre-create updatable mobjects
        x_tracker = ValueTracker(1.5)
        
        y_bar = Line(color="#FF00FF", stroke_width=4)
        y_text = MathTex("y", color="#FF00FF", font_size=24)
        
        formula = MathTex("v = \\sqrt{2gy}", color="#00FFFF", font_size=32)
        self.place_at_grid(formula, "A4", scale_factor=1.0)
        
        # Velocity vector (Arrow)
        v_arrow = Arrow(buff=0, color="#00FFFF", stroke_width=3, max_tip_length_to_length_ratio=0.3)

        # Updaters for movement and dynamic labeling
        def update_bead(m):
            x = x_tracker.get_value()
            m.move_to(np.array([x, wire_func(x), 0]))
            
        def update_y_bar(m):
            x = x_tracker.get_value()
            m.put_start_and_end_on(
                np.array([x - 0.2, 1.2, 0]),
                np.array([x - 0.2, wire_func(x), 0])
            )
            
        def update_y_text(m):
            m.next_to(y_bar, LEFT, buff=0.1)
            
        def update_v_arrow(m):
            x = x_tracker.get_value()
            if x <= 1.51:
                m.set_opacity(0)
                return
            m.set_opacity(1)
            p1 = np.array([x, wire_func(x), 0])
            # Tangent slope for wire_func: y' = -2/3 * (x - 1.5)
            slope = -0.666 * (x - 1.5)
            # Velocity magnitude is proportional to sqrt(y_drop), which is proportional to (x-1.5)
            v_mag = (x - 1.5) * 0.5
            unit_dir = np.array([1, slope, 0])
            unit_dir /= np.linalg.norm(unit_dir)
            m.put_start_and_end_on(p1, p1 + unit_dir * v_mag)
            
        def update_ke_label(m):
            m.next_to(bead, DOWN, buff=0.1)

        # Attach updaters
        bead.add_updater(update_bead)
        y_bar.add_updater(update_y_bar)
        y_text.add_updater(update_y_text)
        v_arrow.add_updater(update_v_arrow)
        ke_label.add_updater(update_ke_label)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.play(
            Create(wire_path),
            FadeIn(bead),
            Write(pe_label),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GREY)
        self.lecture[1].set_color("#FF00FF") # Magenta matches y_bar
        self.play(
            Create(y_bar),
            FadeIn(y_text),
            run_time=1.0
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GREY)
        self.lecture[2].set_color("#00FFFF") # Cyan matches formula and v_arrow
        self.play(
            Write(formula),
            FadeIn(v_arrow),
            FadeIn(ke_label),
            run_time=1.0
        )
        
        # Descent animation
        self.play(
            x_tracker.animate.set_value(4.5),
            run_time=4.0,
            rate_func=rate_functions.ease_in_sine
        )
        
        self.wait(1.5)
