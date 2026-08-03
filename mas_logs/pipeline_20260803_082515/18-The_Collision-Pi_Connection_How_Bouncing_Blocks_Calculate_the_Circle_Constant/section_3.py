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
        self.setup_layout("Prerequisite Knowledge: Conservation Laws", [
            "Kinetic energy is conserved in these elastic collisions.",
            "Momentum also remains conserved during each block impact.",
            "These laws strictly limit the velocities of both blocks."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Equation color: White (#FFFFFF)
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        energy_eq = MathTex(r"\frac{1}{2}mv^2 + \frac{1}{2}MV^2 = E", color=WHITE)
        # Resolved Issue 28: Reposition energy_eq to A2-A5
        self.place_in_area(energy_eq, 'A2', 'A5', scale_factor=1.0)
        self.play(Write(energy_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Momentum line color: Light Green (#00FF00)
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        momentum_eq = MathTex(r"mv + MV = P", color="#00FF00")
        # Resolved Issue 29: Reposition momentum_eq to B2-B5, scale factor 1.0
        self.place_in_area(momentum_eq, 'B2', 'B5', scale_factor=1.0)
        
        # Resolved Issue 30: Energy budget bar repositioned to D3-F3
        bar_bg = Rectangle(height=2.5, width=0.8, color=WHITE, stroke_width=2)
        self.place_in_area(bar_bg, 'D3', 'F3', scale_factor=1.0)
        
        # Solid energy bar representing total constant budget
        energy_bar = Rectangle(height=2.5, width=0.8, fill_color="#00FF00", fill_opacity=0.6, stroke_width=0)
        energy_bar.move_to(bar_bg.get_center())
        
        bar_label = Text("Total Energy Budget", font_size=18, color=WHITE)
        bar_label.next_to(bar_bg, RIGHT, buff=0.3)
        
        self.play(
            Write(momentum_eq),
            Create(bar_bg),
            FadeIn(energy_bar),
            Write(bar_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Resolved Issue 23: Asset Integration
        # Load block assets from /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg
        block_m = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE).scale(0.2)
        block_M = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=RED).scale(0.4)
        
        # Visualizing the shift: split the bar into two segments.
        h_total = 2.5
        w = 0.8
        
        # Persistent mobjects for segments
        e_small = Rectangle(height=0.5, width=w, fill_color=BLUE, fill_opacity=0.8, stroke_width=0)
        e_large = Rectangle(height=2.0, width=w, fill_color=RED, fill_opacity=0.8, stroke_width=0)
        
        # Align segments within bar_bg
        e_small.move_to(bar_bg.get_bottom(), aligned_edge=DOWN)
        e_large.next_to(e_small, UP, buff=0)
        
        e_small_label = MathTex(r"\frac{1}{2}mv^2", font_size=24, color=BLUE)
        e_large_label = MathTex(r"\frac{1}{2}MV^2", font_size=24, color=RED)
        
        # Positioning labels and blocks
        # We group label and block icon together
        group_m = VGroup(block_m, e_small_label).arrange(RIGHT, buff=0.1)
        group_M = VGroup(block_M, e_large_label).arrange(RIGHT, buff=0.1)
        
        # Initial positions relative to bar segments
        group_m.next_to(e_small, LEFT, buff=0.2)
        group_M.next_to(e_large, LEFT, buff=0.2)

        self.play(
            FadeOut(energy_bar),
            FadeIn(e_small),
            FadeIn(e_large),
            FadeIn(group_m),
            FadeIn(group_M)
        )
        self.wait(1)
        
        # Dynamic update logic for the energy shift
        h_val = ValueTracker(0.5) # Initial height of the small block's energy segment
        
        def update_small(m):
            h = h_val.get_value()
            m.stretch_to_fit_height(max(h, 0.01), about_edge=DOWN)
            m.move_to(bar_bg.get_bottom(), aligned_edge=DOWN)
            
        def update_large(m):
            h = h_total - h_val.get_value()
            m.stretch_to_fit_height(max(h, 0.01), about_edge=UP)
            m.move_to(bar_bg.get_top(), aligned_edge=UP)
            
        def update_group_m(m):
            m.next_to(e_small, LEFT, buff=0.2)

        def update_group_M(m):
            m.next_to(e_large, LEFT, buff=0.2)

        e_small.add_updater(update_small)
        e_large.add_updater(update_large)
        group_m.add_updater(update_group_m)
        group_M.add_updater(update_group_M)
        
        # Energy shifts between the blocks
        self.play(h_val.animate.set_value(2.0), run_time=2, rate_func=smooth)
        self.wait(0.5)
        self.play(h_val.animate.set_value(0.2), run_time=2, rate_func=smooth)
        self.wait(1)
        
        # Cleanup updaters before finishing
        e_small.remove_updater(update_small)
        e_large.remove_updater(update_large)
        group_m.remove_updater(update_group_m)
        group_M.remove_updater(update_group_M)
        self.wait(2)
