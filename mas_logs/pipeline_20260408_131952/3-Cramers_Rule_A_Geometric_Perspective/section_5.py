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
        # Setup Script and Layout
        lecture_lines = [
            'Vector B is made of V1 and V2 parts.',
            'The V2 component shears the shape without changing area.',
            "Only the V1 component determines the area's scale."
        ]
        self.setup_layout("Shearing and Scaling Logic", lecture_lines)
        
        # Visual Parameters
        v1_color = BLUE
        v2_color = RED
        b_color = "#00FF00"
        xv1_color = BLUE_A
        yv2_color = PINK
        para_color = YELLOW
        
        # Coordinate Space - Issue 40: Moved axes to B3-F6
        axes = Axes(
            x_range=[-0.5, 3.5, 1], y_range=[-0.5, 3.5, 1],
            x_length=3.5, y_length=3.5,
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, "B3", "F6", scale_factor=0.8)
        origin = axes.c2p(0, 0, 0)
        
        # Vector Data
        v1_val = np.array([1.2, 0.4, 0])
        v2_val = np.array([0.4, 1.3, 0])
        x_coeff = 1.3
        y_coeff = 0.8
        
        xv1_val = x_coeff * v1_val
        yv2_val = y_coeff * v2_val
        b_val = xv1_val + yv2_val

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        v1 = Arrow(origin, axes.c2p(*v1_val), buff=0, color=v1_color)
        v2 = Arrow(origin, axes.c2p(*v2_val), buff=0, color=v2_color)
        v1_lab = Text("V1", color=v1_color, font_size=20)
        self.place_at_grid(v1_lab, "E6", scale_factor=1.0)
        v2_lab = Text("V2", color=v2_color, font_size=20)
        self.place_at_grid(v2_lab, "B3", scale_factor=1.0)
        
        b_vec = Arrow(origin, axes.c2p(*b_val), buff=0, color=b_color)
        b_lab = Text("B", color=b_color, font_size=22)
        self.place_at_grid(b_lab, "B5", scale_factor=1.0)
        
        xv1_line = DashedLine(origin, axes.c2p(*xv1_val), color=xv1_color)
        xv1_lab = Text("xV1", color=xv1_color, font_size=18)
        # Issue 42: xV1 label at E4
        self.place_at_grid(xv1_lab, "E4", scale_factor=0.7)
        
        yv2_line = DashedLine(axes.c2p(*xv1_val), axes.c2p(*b_val), color=yv2_color)
        yv2_lab = Text("yV2", color=yv2_color, font_size=18)
        # Issue 41: yV2 label at C4
        self.place_at_grid(yv2_lab, "C4", scale_factor=0.7)
        
        self.play(Create(axes))
        self.play(Create(v1), Create(v2), Write(v1_lab), Write(v2_lab))
        self.play(Create(b_vec), Write(b_lab))
        self.play(Create(xv1_line), Write(xv1_lab))
        self.play(Create(yv2_line), Write(yv2_lab))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Load Parallelogram Asset - Issue 26
        para_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/parallelogram.svg")
        self.place_at_grid(para_icon, "A4", scale_factor=0.3)
        self.play(FadeIn(para_icon))

        # Original Parallelogram (B, V2)
        p_pts = [origin, axes.c2p(*v2_val), axes.c2p(*(v2_val + b_val)), axes.c2p(*b_val)]
        para = Polygon(*p_pts, fill_opacity=0.3, fill_color=para_color, stroke_color=para_color, stroke_width=2)
        self.play(FadeIn(para))
        
        # Shear tracker: slide B tip back to xV1
        s_val = ValueTracker(1.0)
        
        def update_para(p):
            s = s_val.get_value()
            curr_b = xv1_val + s * yv2_val
            current_pts = [origin, axes.c2p(*v2_val), axes.c2p(*(v2_val + curr_b)), axes.c2p(*curr_b)]
            p.set_points_as_corners([*current_pts, current_pts[0]])
            
        para.add_updater(update_para)
        
        # B vector tip follows the shear
        b_moving = Arrow(origin, axes.c2p(*b_val), buff=0, color=b_color)
        b_moving.add_updater(lambda m: m.put_start_and_end_on(origin, axes.c2p(*(xv1_val + s_val.get_value() * yv2_val))))
        
        # Height indicator relative to V2
        v2_dir = v2_val / np.linalg.norm(v2_val)
        v2_norm = np.array([-v2_dir[1], v2_dir[0], 0])
        h_mag = np.dot(xv1_val, v2_norm)
        h_pt = xv1_val - h_mag * v2_norm
        
        h_line = DashedLine(axes.c2p(*xv1_val), axes.c2p(*h_pt), color=WHITE)
        h_lab = Text("Height", font_size=16, color=WHITE)
        self.place_at_grid(h_lab, "C6", scale_factor=1.0)
        
        base_brace = Brace(Line(origin, axes.c2p(*v2_val)), LEFT, color=v2_color, buff=0.1)
        base_lab = Text("Base", font_size=16, color=v2_color)
        self.place_at_grid(base_lab, "D3", scale_factor=1.0)
        
        self.remove(b_vec, b_lab, yv2_line, yv2_lab)
        self.add(b_moving)
        
        self.play(Create(base_brace), Write(base_lab))
        self.play(Create(h_line), Write(h_lab))
        self.play(s_val.animate.set_value(0), run_time=3)
        
        para.clear_updaters()
        b_moving.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Final static state: Parallelogram is now (xV1, V2)
        # We can fade out construction lines to emphasize the final area
        self.play(FadeOut(base_brace), FadeOut(base_lab), FadeOut(h_line), FadeOut(h_lab), FadeOut(para_icon))
        self.wait(2)
