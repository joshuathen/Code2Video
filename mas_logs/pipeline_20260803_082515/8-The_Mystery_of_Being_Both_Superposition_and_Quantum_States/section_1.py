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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Classical systems exist in one definite state at once.",
            "A light switch is either strictly ON or OFF.",
            "Quantum systems are different; they don't choose sides yet.",
            "Imagine a spinning coin, being both heads and tails.",
            "This 'bothness' is the core of the quantum world."
        ]
        self.setup_layout("The Classical vs. Quantum Divide", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        classical_text = Text("Classical: 0 or 1", font_size=24, color=WHITE)
        # Fix Issue 34: Reposition header
        self.place_in_area(classical_text, 'A2', 'A5')
        
        # Load asset for Issue 31
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg]
        switch_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg", color=WHITE)
        # Fix Issue 35: Reposition switch
        self.place_in_area(switch_asset, 'B3', 'B4', scale_factor=0.6)
        
        self.play(Write(classical_text), Create(switch_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Flip switch multiple times to show binary state
        # Since it's an SVG, we'll use a simple flip animation (e.g., vertical scale change or rotation)
        self.play(switch_asset.animate.rotate(PI, axis=RIGHT), run_time=0.4)
        self.play(switch_asset.animate.rotate(PI, axis=RIGHT), run_time=0.4)
        self.play(switch_asset.animate.rotate(PI, axis=RIGHT), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        quantum_text = Text("Quantum: Both?", font_size=24, color="#00FFFF")
        # Fix Issue 36: Reposition quantum header
        self.place_in_area(quantum_text, 'D2', 'D5')
        
        self.play(Write(quantum_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Coin setup
        coin = Circle(radius=0.7, color=WHITE, fill_opacity=0.3)
        self.place_in_area(coin, 'E3', 'F4')
        
        heads_label = Text("H", font_size=36, color=WHITE)
        tails_label = Text("T", font_size=36, color=WHITE)
        heads_label.move_to(coin.get_center())
        tails_label.move_to(coin.get_center())
        
        h_side_label = Text("Heads", font_size=18, color=WHITE)
        t_side_label = Text("Tails", font_size=18, color=WHITE)
        self.place_at_grid(h_side_label, 'E2')
        self.place_at_grid(t_side_label, 'F5')

        # Tracker for rotation
        spin_tracker = ValueTracker(0)
        
        # Group for the coin and its internal labels
        coin_group = VGroup(coin, heads_label, tails_label)
        
        # Updater for the spinning coin (visualizing superposition)
        def update_coin(m):
            val = spin_tracker.get_value()
            # Oscillation of width to simulate spin
            scale_x = np.cos(val * 2 * PI)
            # Avoid too-thin objects or zero scale
            w_factor = max(0.05, abs(scale_x))
            m[0].stretch_to_fit_width(w_factor * 1.4)
            
            # Switch between showing H and T
            if scale_x > 0:
                m[1].set_opacity(1)
                m[2].set_opacity(0)
                m[1].stretch_to_fit_width(max(0.05, abs(scale_x) * 0.8))
            else:
                m[1].set_opacity(0)
                m[2].set_opacity(1)
                m[2].stretch_to_fit_width(max(0.05, abs(scale_x) * 0.8))

        coin_group.add_updater(update_coin)
        
        self.add(coin_group)
        self.play(
            spin_tracker.animate.set_value(4), 
            FadeIn(h_side_label), 
            FadeIn(t_side_label),
            run_time=3, 
            rate_func=linear
        )
        
        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Glow effect
        glow = Circle(radius=1.0, color="#00FFFF", fill_opacity=0.15, stroke_width=0)
        glow.move_to(coin.get_center())
        
        def update_glow(m):
            # Pulsing effect
            pulse = np.sin(spin_tracker.get_value() * 3)
            m.set_fill(opacity=0.1 + 0.1 * abs(pulse))
            # Fixed scale updating logic to avoid recursive scaling
            target_width = 1.6 + 0.2 * pulse
            m.width = target_width
            m.height = target_width

        glow.add_updater(update_glow)
        
        self.add(glow)
        # Continue spinning and pulsing
        self.play(spin_tracker.animate.set_value(10), run_time=4, rate_func=linear)
        
        self.wait(2)
