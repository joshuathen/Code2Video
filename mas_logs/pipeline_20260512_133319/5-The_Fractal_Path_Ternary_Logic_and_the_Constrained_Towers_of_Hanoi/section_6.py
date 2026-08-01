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
        # Colors
        TITLE_COLOR = "#FFD700"  # Golden glowing
        L1_COLOR = "#A0DFFF"     # Light blue
        L2_COLOR = "#D4FFD4"     # Light green
        L3_COLOR = "#FFD4FF"     # Light magenta
        
        self.setup_layout(
            "Conclusion: The Beauty of Mathematical Convergence", 
            [
                "Ternary logic and fractals reveal the same underlying truth.", 
                "Simple recursive rules build infinite, self-similar geometric patterns.", 
                "Counting, graphs, and geometry converge in this restricted puzzle."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(L1_COLOR)
        
        # Helper to generate Sierpinski triangle vertices and edges
        def get_sierpinski_points(order, center, side_length):
            height = (3**0.5 / 2) * side_length
            top = center + UP * (height / 2)
            left = center + LEFT * (side_length / 2) + DOWN * (height / 2)
            right = center + RIGHT * (side_length / 2) + DOWN * (height / 2)
            
            if order == 0:
                return [top]
            
            new_side = side_length / 2
            h_offset = height / 4
            w_offset = side_length / 4
            
            pts_top = get_sierpinski_points(order - 1, center + UP * h_offset, new_side)
            pts_left = get_sierpinski_points(order - 1, center + DOWN * h_offset + LEFT * w_offset, new_side)
            pts_right = get_sierpinski_points(order - 1, center + DOWN * h_offset + RIGHT * w_offset, new_side)
            
            return pts_top + pts_left + pts_right

        # Generate N=2 gasket vertices and labels (9 states)
        def get_ternary_label(index, order):
            res = ""
            for _ in range(order):
                res = str(index % 3) + res
                index //= 3
            return res

        gasket_center = self.grid["D4"]
        n2_points = get_sierpinski_points(2, gasket_center, 4.0)
        
        # Create visual nodes and labels
        nodes = VGroup()
        labels = VGroup()
        for i, pt in enumerate(n2_points):
            dot = Dot(pt, radius=0.08, color=L1_COLOR)
            lbl = Text(get_ternary_label(i, 2), font_size=14, color=WHITE).next_to(dot, UP, buff=0.1)
            nodes.add(dot)
            labels.add(lbl)

        # Create basic triangle outline for N=2
        edges = VGroup()
        def create_triangle_lines(p1, p2, p3):
            return VGroup(Line(p1, p2, stroke_width=2), Line(p2, p3, stroke_width=2), Line(p3, p1, stroke_width=2))

        # Hardcoded triangles for N=2 visually
        h = (3**0.5 / 2) * 4.0
        p_top = gasket_center + UP * (h / 2)
        p_left = gasket_center + LEFT * 2 + DOWN * (h / 2)
        p_right = gasket_center + RIGHT * 2 + DOWN * (h / 2)
        p_mid_tl = (p_top + p_left) / 2
        p_mid_tr = (p_top + p_right) / 2
        p_mid_b = (p_left + p_right) / 2
        
        edges.add(create_triangle_lines(p_top, p_mid_tl, p_mid_tr))
        edges.add(create_triangle_lines(p_left, p_mid_tl, p_mid_b))
        edges.add(create_triangle_lines(p_right, p_mid_tr, p_mid_b))
        edges.set_color(L1_COLOR).set_stroke(opacity=0.5)

        self.play(Create(edges), run_time=1.5)
        self.play(FadeIn(nodes), Write(labels), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(L2_COLOR)

        # Recursive function to generate lines for higher-order Gasket
        def get_gasket_lines(order, center, side):
            if order == 1:
                h = (3**0.5 / 2) * side
                t = center + UP * (h / 2)
                l = center + LEFT * (side / 2) + DOWN * (h / 2)
                r = center + RIGHT * (side / 2) + DOWN * (h / 2)
                return VGroup(Line(t, l), Line(l, r), Line(r, t))
            
            new_side = side / 2
            h = (3**0.5 / 2) * side
            h_off = h / 4
            w_off = side / 4
            
            g1 = get_gasket_lines(order - 1, center + UP * h_off, new_side)
            g2 = get_gasket_lines(order - 1, center + DOWN * h_off + LEFT * w_off, new_side)
            g3 = get_gasket_lines(order - 1, center + DOWN * h_off + RIGHT * w_off, new_side)
            return VGroup(g1, g2, g3)

        # Build N=6 Gasket
        n6_gasket = get_gasket_lines(6, gasket_center, 4.0)
        n6_gasket.set_stroke(width=1, color=L2_COLOR)

        # Hide L1 elements and "zoom out" by scaling down N=2 and replacing with N=6
        self.play(
            FadeOut(labels),
            FadeOut(nodes),
            FadeOut(edges),
            run_time=1
        )
        
        # Show growth of fractal complexity
        self.play(FadeIn(n6_gasket), run_time=1)
        
        # Simulating zoom out: scale the gasket
        self.play(
            n6_gasket.animate.scale(0.5).move_to(self.grid["D4"]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(L3_COLOR)

        # Glowing Final Title
        final_title = Text("The Fractal Path: Ternary Logic", font_size=36)
        final_title.set_color_by_gradient(TITLE_COLOR, "#FFFACD", TITLE_COLOR)
        
        # Position at bottom area to avoid obstructing the fractal (Issue 41)
        self.place_in_area(final_title, "E1", "F6", scale_factor=0.7)
        
        # Make it "glow" with a shadow/stroke copy
        glow = final_title.copy().set_stroke(TITLE_COLOR, width=8, opacity=0.3)
        
        self.play(
            n6_gasket.animate.set_stroke(opacity=0.2),
            Write(final_title),
            FadeIn(glow),
            run_time=2
        )
        self.wait(3)
