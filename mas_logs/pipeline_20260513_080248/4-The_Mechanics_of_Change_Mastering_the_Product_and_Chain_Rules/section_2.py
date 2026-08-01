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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'The product rule handles two functions multiplied together.',
            'Imagine a rectangle with expanding side lengths u and v.',
            'The area change depends on both growth rates.',
            "It's u times v-prime plus v times u-prime.",
            'Robots use this to calculate total planting rates.'
        ]
        self.setup_layout("The Product Rule: The Expanding Rectangle", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        product_label = Text("f(x) * g(x)", font_size=36, color=WHITE)
        # Issue 25: Adjust area to 'B2' to 'B5' to avoid vertical bloat
        self.place_in_area(product_label, "B2", "B5", scale_factor=1.0)
        self.play(Write(product_label))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        u_color = "#FF0000"
        v_color = "#0000FF"
        
        # Base Rectangle (u by v)
        base_rect = Rectangle(width=2.0, height=1.5, stroke_color=WHITE, stroke_width=2)
        u_label = Text("u", color=u_color, font_size=24)
        v_label = Text("v", color=v_color, font_size=24)
        
        # Position base group at grid D3
        rect_origin = VGroup(base_rect)
        self.place_at_grid(rect_origin, "D3")
        
        # Anchor labels to rectangle (using manual next_to relative to the placed rect)
        u_label.next_to(base_rect, DOWN, buff=0.1)
        v_label.next_to(base_rect, LEFT, buff=0.1)
        
        self.play(
            FadeOut(product_label),
            Create(base_rect),
            Write(u_label),
            Write(v_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        du_color = "#FF7F7F" # Light Red
        dv_color = "#7F7FFF" # Light Blue
        
        # Expansion du (width growth)
        v_du_rect = Rectangle(width=0.6, height=1.5, fill_color=du_color, fill_opacity=0.6, stroke_width=1)
        v_du_rect.next_to(base_rect, RIGHT, buff=0)
        du_label = Text("du", color=du_color, font_size=20)
        du_label.next_to(v_du_rect, DOWN, buff=0.1)
        
        # Expansion dv (height growth)
        u_dv_rect = Rectangle(width=2.0, height=0.5, fill_color=dv_color, fill_opacity=0.6, stroke_width=1)
        u_dv_rect.next_to(base_rect, UP, buff=0)
        dv_label = Text("dv", color=dv_color, font_size=20)
        dv_label.next_to(u_dv_rect, LEFT, buff=0.1)
        
        # Negligible corner
        corner = Rectangle(width=0.6, height=0.5, fill_color=GREY, fill_opacity=0.3, stroke_width=1)
        corner.next_to(v_du_rect, UP, buff=0)

        self.play(
            FadeIn(v_du_rect),
            Write(du_label),
            FadeIn(u_dv_rect),
            Write(dv_label),
            FadeIn(corner)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Labels for areas
        area_v_du = Text("v * du", color=du_color, font_size=20).move_to(v_du_rect.get_center())
        area_u_dv = Text("u * dv", color=dv_color, font_size=20).move_to(u_dv_rect.get_center())
        
        # Main Formula
        formula = Text("(u * v)' = u * v' + v * u'", color=WHITE, font_size=32)
        # Issue 26: Change point-based placement to area-based for centering
        self.place_in_area(formula, "B2", "B5", scale_factor=0.9)
        
        self.play(
            Write(area_v_du),
            Write(area_u_dv),
            Write(formula)
        )
        
        # Highlight logic
        self.play(
            v_du_rect.animate.set_stroke(color=YELLOW, width=4),
            u_dv_rect.animate.set_stroke(color=YELLOW, width=4)
        )
        self.play(
            v_du_rect.animate.set_stroke(color=WHITE, width=1),
            u_dv_rect.animate.set_stroke(color=WHITE, width=1)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        robot_formula = Text("Rate = (Arms * Speed') + (Speed * Arms')", font_size=20, color=WHITE)
        # Issue 27: Use area placement to avoid clipping and improve alignment
        self.place_in_area(robot_formula, "F1", "F6", scale_factor=0.7)
        
        self.play(Write(robot_formula))
        self.wait(3)
