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
        
        # --- Visualization Elements ---
        # 1. The Wire Path (B2 to E5)
        start_pt = self.grid["B2"]
        end_pt = self.grid["E5"]
        # Midpoint-based control points for a smooth curve
        cp1 = self.grid["B4"]
        cp2 = self.grid["C5"]
        wire = CubicBezier(start_pt, cp1, cp2, end_pt, color="#888888")
        
        # 2. The Bead
        bead = Dot(color="#FFFFFF", radius=0.1)
        bead.move_to(start_pt)
        
        # 3. Height indicator 'y'
        # Pre-create y_bar and y_label to be updated in sync_scene
        y_bar = Line(start_pt, start_pt, color="#FF00FF", stroke_width=4)
        y_label = MathTex("y", color="#FF00FF", font_size=28)
        
        # 4. Physics formula
        formula = MathTex(r"v = \sqrt{2gy}", color="#00FFFF", font_size=32)
        self.place_at_grid(formula, "A5", scale_factor=1.0)
        
        # 5. Velocity arrow
        # Initial dummy arrow, will be updated in sync_scene
        v_arrow = Arrow(start_pt, start_pt + RIGHT, color="#00FFFF", buff=0, 
                        stroke_width=3, max_tip_length_to_length_ratio=0.2)
        v_arrow.set_opacity(0)

        # --- Control Logic ---
        # ValueTracker controls the proportion along the curve
        alpha = ValueTracker(0)

        def sync_scene(m):
            a = alpha.get_value()
            # Calculate current position on wire once per frame
            p = wire.point_from_proportion(a)
            
            # Update Bead position
            bead.move_to(p)
            
            # Update Vertical Bar (from initial y-level to current position)
            y_top = np.array([p[0], start_pt[1], 0])
            y_bar.put_start_and_end_on(y_top, p)
            
            # Update 'y' label (placed left of the vertical bar's midpoint)
            y_label.move_to(y_top + DOWN * (start_pt[1] - p[1]) / 2 + LEFT * 0.3)
            
            # Update Velocity Arrow
            if a > 0.005:
                v_arrow.set_opacity(1)
                # Approximate tangent by looking back a tiny bit
                p_prev = wire.point_from_proportion(a - 0.005)
                tangent = p - p_prev
                mag = np.linalg.norm(tangent)
                if mag > 1e-5:
                    unit_t = tangent / mag
                    # Physical relation: v is proportional to sqrt(y_drop)
                    y_drop = max(0, start_pt[1] - p[1])
                    v_len = np.sqrt(y_drop) * 1.2 
                    v_arrow.put_start_and_end_on(p, p + unit_t * v_len)
            else:
                v_arrow.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        # "In gravity, potential energy converts to kinetic energy."
        self.lecture[0].set_color(WHITE)
        self.play(Create(wire), FadeIn(bead), run_time=1.2)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "Velocity depends on the vertical distance fallen."
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FF00FF")
        
        # Activate the shared updater on the bead and add height elements
        bead.add_updater(sync_scene)
        self.add(y_bar, y_label)
        
        # Small descent to visually introduce the concept of 'y'
        self.play(alpha.animate.set_value(0.15), run_time=1)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # "This links depth directly to the speed of travel."
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#00FFFF")
        
        self.play(FadeIn(formula), FadeIn(v_arrow), run_time=0.8)
        
        # Complete the full descent descent
        self.play(alpha.animate.set_value(1.0), run_time=3.5, rate_func=slow_into)
        self.wait(2.0)
        
        # Cleanup updaters to prevent unexpected behavior in potential sequence
        bead.clear_updaters()
