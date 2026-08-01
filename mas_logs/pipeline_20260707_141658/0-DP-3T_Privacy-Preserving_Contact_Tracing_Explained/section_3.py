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

class Section3Scene(Scene):
    def construct(self):
        # Configuration for visual elements
        font_size_val = 32
        radius_val = 0.6
        circle_color = BLUE
        highlight_color = YELLOW

        # --- Create SK_{t-1} ---
        sk_prev_circle = Circle(radius=radius_val, color=circle_color)
        # Replaced MathTex with Text to avoid the 'latex' dependency error
        sk_prev_label = Text("SK_{t-1}", font_size=font_size_val)
        sk_prev_group = VGroup(sk_prev_circle, sk_prev_label)

        # --- Create SK_{t} ---
        sk_curr_circle = Circle(radius=radius_val, color=circle_color)
        # Replaced MathTex with Text to avoid the 'latex' dependency error
        sk_curr_label = Text("SK_{t}", font_size=font_size_val)
        sk_curr_group = VGroup(sk_curr_circle, sk_curr_label)

        # --- Create SK_{t+1} ---
        sk_next_circle = Circle(radius=radius_val, color=circle_color)
        # Replaced MathTex with Text to avoid the 'latex' dependency error
        sk_next_label = Text("SK_{t+1}", font_size=font_size_val)
        sk_next_group = VGroup(sk_next_circle, sk_next_label)

        # Arrange groups horizontally
        chain_vgroup = VGroup(sk_prev_group, sk_curr_group, sk_next_group)
        chain_vgroup.arrange(RIGHT, buff=1.5)

        # --- Create Connection Arrows ---
        arrow1 = Arrow(
            start=sk_prev_group.get_right(),
            end=sk_curr_group.get_left(),
            buff=0.15,
            color=WHITE,
            stroke_width=3
        )
        arrow2 = Arrow(
            start=sk_curr_group.get_right(),
            end=sk_next_group.get_left(),
            buff=0.15,
            color=WHITE,
            stroke_width=3
        )

        # Group everything for the scene
        full_chain = VGroup(sk_prev_group, arrow1, sk_curr_group, arrow2, sk_next_group)
        full_chain.move_to(ORIGIN)

        # --- Animation Sequence ---
        # 1. Fade in the whole chain
        self.play(FadeIn(full_chain, shift=UP))
        self.wait(1)

        # 2. Highlight the current key SK_t
        self.play(
            sk_curr_circle.animate.set_color(highlight_color).set_stroke(width=6),
            sk_curr_label.animate.scale(1.2).set_color(highlight_color),
            run_time=1
        )
        self.wait(0.5)

        # 3. Emphasis on the derivation (Arrows pulse)
        self.play(
            arrow1.animate.set_color(highlight_color),
            arrow2.animate.set_color(highlight_color),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        self.wait(2)
