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

        # Asset loading: Render the bridge structure [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/bridge.svg]
        bridge = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/bridge.svg")
        bridge.set_stroke(color=COLOR_BRIDGE, width=2)
        bridge.set_fill(opacity=0)
        # Position bridge in the central-right area
        self.place_in_area(bridge, "C2", "E6", scale_factor=1.5)
        
        # Capture positioned points to allow consistent deformation
        original_points = bridge.points.copy()
        
        # Trackers for vibration animation
        time_tracker = ValueTracker(0)
        amp_tracker = ValueTracker(0.0) # Start with no vibration
        freq_tracker = ValueTracker(1.0)

        # Vibration updater: Animate the bridge structure [Asset: ...] vibrating in a mode defined by its eigenvector.
        def update_bridge(mob):
            t = time_tracker.get_value()
            A = amp_tracker.get_value()
            f = freq_tracker.get_value()
            
            pts = original_points.copy()
            x_vals = pts[:, 0]
            if len(x_vals) == 0: return
            
            x_min, x_max = np.min(x_vals), np.max(x_vals)
            L = x_max - x_min if x_max > x_min else 1.0
            
            for i in range(len(pts)):
                x = pts[i][0]
                # Fundamental mode shape: sin(pi * x / L)
                y_disp = A * np.sin(np.pi * (x - x_min) / L) * np.sin(2 * np.pi * f * t)
                pts[i][1] += y_disp
            mob.set_points(pts)

        bridge.add_updater(update_bridge)
        
        # Labels - Fix positions per Issue 47
        lambda_label = Text("λ = Frequency", color=COLOR_EIGVAL, font_size=28)
        # Issue 47 Fix: lambda_label at B4
        self.place_at_grid(lambda_label, "B4", scale_factor=0.8)
        
        v_label = Text("v = Mode Shape", color=COLOR_EIGVEC, font_size=28)
        self.place_at_grid(v_label, "B2", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_EIGVAL)
        self.add(bridge)
        self.play(Write(lambda_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_EIGVEC)
        self.play(Write(v_label))
        
        # Start small vibration
        self.add_updater(lambda dt: time_tracker.increment_value(dt))
        self.play(amp_tracker.animate.set_value(0.2), run_time=2)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_RESONANCE)
        
        # Resonance visualization (High amplitude)
        # Issue 48 Fix: res_text at D4 with scale 0.7
        res_text = Text("RESONANCE!", color=COLOR_RESONANCE, font_size=36)
        self.place_at_grid(res_text, "D4", scale_factor=0.7)
        
        self.play(
            amp_tracker.animate.set_value(0.8),
            FadeIn(res_text),
            run_time=1.5
        )
        self.play(Indicate(res_text, color=COLOR_RESONANCE))
        self.wait(1)
        
        # Show stabilization (Avoiding resonance)
        # Issue 49 Fix: safe_text using area positioning E3 to E5
        safe_text = Text("Safe Frequency", color=WHITE, font_size=24)
        self.place_in_area(safe_text, "E3", "E5", scale_factor=0.8)
        
        self.play(
            amp_tracker.animate.set_value(0.05),
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
            FadeOut(bridge)
        )
