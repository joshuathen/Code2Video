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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "Linear Independence: The Essential Team"
        lines = [
            'Independent vectors provide unique, non-redundant directions.',
            'Each vector expands our span to a higher dimension.',
            'These are the essential moves needed to explore space.'
        ]
        self.setup_layout(title, lines)

        # Vector Definitions
        # Using specific coordinates to ensure they look independent and 3D-perspective friendly in 2D
        p_color = "#7B68EE"
        q_color = "#FFDAB9"
        r_color = "#ADFF2F"
        
        origin_pt = ORIGIN
        p_vec = np.array([1.5, 0.2, 0])
        q_vec = np.array([0.5, 1.5, 0])
        r_vec = np.array([-1.2, 0.8, 0])

        # === Animation for Lecture Line 1 ===
        # Independent vectors provide unique, non-redundant directions.
        
        vec_p = Arrow(origin_pt, p_vec, color=p_color, buff=0)
        # Replaced MathTex with Text to avoid LaTeX dependency error
        label_p = Text("p", color=p_color, font_size=24, slant=ITALIC).next_to(vec_p.get_end(), RIGHT, buff=0.1)
        
        # 1D Span: A line through origin
        span_1d = Line(p_vec * -2, p_vec * 2, color=p_color, stroke_opacity=0.3)
        
        group_1d = VGroup(span_1d, vec_p, label_p)
        self.place_in_area(group_1d, 'B2', 'E5')

        self.play(
            self.lecture[0].animate.set_color(p_color),
            FadeIn(span_1d),
            GrowArrow(vec_p),
            Write(label_p),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Each vector expands our span to a higher dimension.
        
        vec_q = Arrow(origin_pt, q_vec, color=q_color, buff=0)
        # Replaced MathTex with Text to avoid LaTeX dependency error
        label_q = Text("q", color=q_color, font_size=24, slant=ITALIC).next_to(vec_q.get_end(), UP, buff=0.1)
        
        # 2D Span: A parallelogram (plane)
        span_2d = Polygon(
            (p_vec + q_vec) * -1.5,
            (p_vec - q_vec) * 1.5,
            (p_vec + q_vec) * 1.5,
            (q_vec - p_vec) * 1.5,
            fill_color=q_color,
            fill_opacity=0.2,
            stroke_width=0
        )
        
        # Group everything and re-place to maintain central origin
        vis_group = VGroup(span_1d, span_2d, vec_p, vec_q, label_p, label_q)
        self.place_in_area(vis_group, 'B2', 'E5')

        self.play(
            self.lecture[1].animate.set_color(q_color),
            FadeIn(span_2d),
            GrowArrow(vec_q),
            Write(label_q),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # These are the essential moves needed to explore space.
        
        vec_r = Arrow(origin_pt, r_vec, color=r_color, buff=0)
        # Replaced MathTex with Text to avoid LaTeX dependency error
        label_r = Text("r", color=r_color, font_size=24, slant=ITALIC).next_to(vec_r.get_end(), LEFT, buff=0.1)
        
        # Re-center with vector r
        final_vis_group = VGroup(span_1d, span_2d, vec_p, vec_q, vec_r, label_p, label_q, label_r)
        self.place_in_area(final_vis_group, 'B2', 'E5')

        self.play(
            self.lecture[2].animate.set_color(r_color),
            GrowArrow(vec_r),
            Write(label_r),
            run_time=1.5
        )
        self.wait(0.5)

        # Pulse p, q, and r sequentially in white (#FFFFFF)
        for v in [vec_p, vec_q, vec_r]:
            self.play(Indicate(v, color=WHITE, scale_factor=1.2), run_time=0.8)
        
        self.wait(1)

        # Show no vector is a combination of the others
        combo_p = vec_p.copy().set_color(WHITE).set_stroke(opacity=0.5)
        combo_q = vec_q.copy().set_color(WHITE).set_stroke(opacity=0.5).shift(p_vec)
        sum_vec = Arrow(origin_pt, p_vec + q_vec, color=WHITE, stroke_width=2, buff=0)
        # Replaced MathTex with Text and used Unicode for the inequality sign
        not_equal = Text("r ≠ ap + bq", font_size=24, color=WHITE, slant=ITALIC)
        self.place_at_grid(not_equal, 'F4')

        self.play(
            Create(combo_p),
            run_time=0.7
        )
        self.play(
            Create(combo_q),
            run_time=0.7
        )
        self.play(
            Create(sum_vec),
            Write(not_equal),
            run_time=1
        )
        
        # Final highlight of independence
        self.play(
            vec_r.animate.set_color(WHITE),
            sum_vec.animate.set_color(RED),
            run_time=1
        )
        self.play(
            vec_r.animate.set_color(r_color),
            FadeOut(combo_p),
            FadeOut(combo_q),
            FadeOut(sum_vec),
            FadeOut(not_equal),
            run_time=1
        )
        
        self.wait(2)