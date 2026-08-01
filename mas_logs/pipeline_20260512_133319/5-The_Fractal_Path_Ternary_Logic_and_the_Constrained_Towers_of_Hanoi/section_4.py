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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initialize Scene
        lecture_lines = [
            'Valid moves connect states into a geometric graph.',
            'One disk creates a simple three-node linear path.',
            'Two disks form three connected lines in a triangle.',
            'Three disks expand this structure into a larger triangle.',
            'The resulting state graph forms a beautiful Sierpinski Gasket.'
        ]
        self.setup_layout("The Geometry of Moves: The Sierpinski Gasket", lecture_lines)

        # Colors for each stage
        colors = ["#88C0D0", "#A3BE8C", "#EBCB8B", "#D08770", "#BF616A"]

        # Helper to build the 9-vertex graph
        def make_9_vertex_graph(color):
            h_big, s_big = 1.8, 2.0
            p_top = np.array([0, h_big/2, 0])
            p_bl = np.array([-s_big/2, -h_big/2, 0])
            p_br = np.array([s_big/2, -h_big/2, 0])
            s_sub, h_sub = s_big / 2, h_big / 2
            c1 = p_top + np.array([0, -h_sub/3, 0])
            c2 = p_bl + np.array([s_sub/4, h_sub/3, 0])
            c3 = p_br + np.array([-s_sub/4, h_sub/3, 0])
            
            def get_tri(center, color):
                pts = [center + np.array([0, h_sub/3, 0]), 
                       center + np.array([-s_sub/4, -h_sub/6, 0]), 
                       center + np.array([s_sub/4, -h_sub/6, 0])]
                dots = VGroup(*[Dot(p, radius=0.06, color=WHITE) for p in pts])
                lines = VGroup(Line(pts[0], pts[1], color=color), 
                               Line(pts[1], pts[2], color=color), 
                               Line(pts[2], pts[0], color=color))
                return VGroup(lines, dots)
            
            g1, g2, g3 = get_tri(c1, color), get_tri(c2, color), get_tri(c3, color)
            link12 = Line(g1[1][1].get_center(), g2[1][0].get_center(), color=color)
            link23 = Line(g2[1][2].get_center(), g3[1][1].get_center(), color=color)
            link31 = Line(g3[1][0].get_center(), g1[1][2].get_center(), color=color)
            return VGroup(g1, g2, g3, link12, link23, link31)

        # Helper to build the 27-vertex graph
        def make_27_vertex_graph(color):
            sub = make_9_vertex_graph(color).scale(0.45)
            h, w = 1.0, 1.1
            g1 = sub.copy().move_to(UP * h)
            g2 = sub.copy().move_to(DOWN * 0.5 + LEFT * w)
            g3 = sub.copy().move_to(DOWN * 0.5 + RIGHT * w)
            return VGroup(g1, g2, g3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        one_disk_nodes = VGroup(*[Dot(radius=0.1, color=WHITE) for _ in range(3)]).arrange(RIGHT, buff=1.0)
        one_disk_edges = VGroup(
            Line(one_disk_nodes[0].get_center(), one_disk_nodes[1].get_center(), color=colors[0]),
            Line(one_disk_nodes[1].get_center(), one_disk_nodes[2].get_center(), color=colors[0])
        )
        one_disk_path = VGroup(one_disk_edges, one_disk_nodes)
        self.place_in_area(one_disk_path, 'A4', 'A6', scale_factor=0.8)
        self.play(Create(one_disk_path))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        path_copy1 = one_disk_path.copy().scale(0.5).shift(UP*0.8)
        path_copy2 = one_disk_path.copy().scale(0.5).shift(DOWN*0.4 + LEFT*0.8)
        path_copy3 = one_disk_path.copy().scale(0.5).shift(DOWN*0.4 + RIGHT*0.8)
        paths_vgroup = VGroup(path_copy1, path_copy2, path_copy3)
        self.place_in_area(paths_vgroup, 'B4', 'C6', scale_factor=0.9)
        self.play(ReplacementTransform(one_disk_path, paths_vgroup))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        two_disks_graph = make_9_vertex_graph(colors[2])
        self.place_in_area(two_disks_graph, 'B4', 'C6', scale_factor=0.9)
        self.play(ReplacementTransform(paths_vgroup, two_disks_graph))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        sierpinski_gasket = make_27_vertex_graph(colors[3])
        self.place_in_area(sierpinski_gasket, 'D1', 'F6', scale_factor=0.7)
        self.play(ReplacementTransform(two_disks_graph, sierpinski_gasket))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.play(sierpinski_gasket.animate.set_color(colors[4]))
        self.wait(2)
