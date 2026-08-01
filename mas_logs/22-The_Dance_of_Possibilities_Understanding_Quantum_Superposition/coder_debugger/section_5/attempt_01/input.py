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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initialize Scene
        lecture_lines = [
            'Measuring a system forces its superposition to vanish.',
            'An observer looking causes the state to collapse.',
            'The system instantly snaps into one definite classical state.'
        ]
        self.setup_layout("The Collapse: The Act of Measurement", lecture_lines)

        # Elements setup
        # 1. Coin [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/coin.svg]
        coin = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/coin.svg")
        coin.set_color("#888888")
        # Issue 39 Fix: Relocate coin to D3-E4
        self.place_in_area(coin, "D3", "E4", scale_factor=0.9)
        
        # 2. Eye [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/eye.svg]
        eye = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/eye.svg")
        eye.set_color(WHITE)
        # Issue 38 Fix: Relocate eye to B3-B4
        self.place_in_area(eye, "B3", "B4", scale_factor=0.6)
        
        # 3. Heads State (Coin SVG + Label)
        heads_coin = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/coin.svg")
        heads_coin.set_color("#00FF00")
        heads_label = Text("Heads", font_size=24, color="#00FF00")
        heads_group = VGroup(heads_coin, heads_label).arrange(DOWN, buff=0.1)
        # Issue 40 Fix: Relocate heads_group to D3-E4
        self.place_in_area(heads_group, "D3", "E4", scale_factor=0.9)

        # Tracker for "spinning" effect
        spin_tracker = ValueTracker(0)
        initial_width = coin.width
        
        def update_coin(m):
            # Oscillate width to simulate 3D rotation in 2D
            val = spin_tracker.get_value()
            new_width = initial_width * abs(np.cos(val * PI))
            # Ensure width is at least very small to avoid zero-division errors
            m.stretch_to_fit_width(max(new_width, 0.01))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(coin))
        coin.add_updater(update_coin)
        self.play(spin_tracker.animate.set_value(4), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        self.play(FadeIn(eye, shift=DOWN))
        self.play(spin_tracker.animate.set_value(8), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Stop spinning instantly and "collapse"
        coin.remove_updater(update_coin)
        
        # Create a flash effect to represent measurement impact
        flash = Flash(coin.get_center(), color=WHITE, line_length=0.5)
        
        self.play(
            FadeOut(coin),
            FadeOut(eye),
            FadeIn(heads_group),
            flash
        )
        
        self.wait(3)

        # Cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
