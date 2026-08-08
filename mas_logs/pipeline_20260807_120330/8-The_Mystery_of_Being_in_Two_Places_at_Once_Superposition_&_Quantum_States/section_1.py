from manim import *

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
        # === Setup ===
        title_text = "The Hook: The Spinning Coin"
        lecture_lines = [
            "In our world, a coin is Heads or Tails.",
            "Quantum coins are like coins spinning on tables.",
            "They represent both possible outcomes simultaneously."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        HEADS_COLOR = "#FFD700"  # Gold
        TAILS_COLOR = "#C0C0C0"  # Silver
        QUANTUM_COLOR = "#00FFFF" # Cyan
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HEADS_COLOR))
        
        # Heads Coin - Fixed position per Issue 37
        heads_coin = Circle(radius=0.45, color=HEADS_COLOR, fill_opacity=0.6)
        heads_label = Text("H", color=HEADS_COLOR).scale(0.8)
        heads_group = VGroup(heads_coin, heads_label)
        self.place_at_grid(heads_group, "B3", scale_factor=0.8)
        
        # Tails Coin - Fixed position per Issue 38
        tails_coin = Circle(radius=0.45, color=TAILS_COLOR, fill_opacity=0.6)
        tails_label = Text("T", color=TAILS_COLOR).scale(0.8)
        tails_group = VGroup(tails_coin, tails_label)
        self.place_at_grid(tails_group, "B4", scale_factor=0.8)
        
        self.play(FadeIn(heads_group), FadeIn(tails_group))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(QUANTUM_COLOR)
        )
        
        # Prepare spinning coin - Fixed area per Issue 39
        spinning_base = Circle(radius=0.5, color=WHITE, fill_opacity=0.1)
        self.place_in_area(spinning_base, 'C3', 'D4', scale_factor=1.0)
        
        width_tracker = ValueTracker(0.0)
        
        # Use always_redraw for the ellipse to simulate rotation
        spinning_visual = always_redraw(lambda: Ellipse(
            width=max(0.01, abs(np.cos(width_tracker.get_value() * PI))),
            height=1.0,
            color=WHITE,
            stroke_width=2
        ).move_to(spinning_base.get_center()))
        
        self.play(
            FadeOut(heads_group),
            FadeOut(tails_group),
            FadeIn(spinning_base),
            FadeIn(spinning_visual)
        )
        
        # Spin the coin
        self.play(width_tracker.animate.set_value(4), run_time=2, rate_func=linear)
        
        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(QUANTUM_COLOR)
        )
        
        # Flashing H and T - use persistent objects with updaters
        h_flash = Text("H", color=HEADS_COLOR).scale(0.8).move_to(spinning_base.get_center())
        t_flash = Text("T", color=TAILS_COLOR).scale(0.8).move_to(spinning_base.get_center())
        
        # Visibility logic based on width_tracker
        h_flash.add_updater(lambda m: m.set_opacity(1 if (int(width_tracker.get_value() * 4) % 2 == 0) else 0))
        t_flash.add_updater(lambda m: m.set_opacity(1 if (int(width_tracker.get_value() * 4) % 2 != 0) else 0))
        
        # Blur effect (glow)
        blur_glow = Circle(radius=0.7, color=QUANTUM_COLOR, fill_opacity=0.15, stroke_width=0)
        blur_glow.move_to(spinning_base.get_center())
        
        self.add(h_flash, t_flash)
        
        # Speed up spin and flash
        self.play(
            width_tracker.animate.set_value(10),
            FadeIn(blur_glow),
            run_time=3, 
            rate_func=linear
        )
        
        # Clean up updaters for stable end state
        h_flash.clear_updaters()
        t_flash.clear_updaters()
        h_flash.set_opacity(0.5)
        t_flash.set_opacity(0.5)
        
        self.wait(2)
