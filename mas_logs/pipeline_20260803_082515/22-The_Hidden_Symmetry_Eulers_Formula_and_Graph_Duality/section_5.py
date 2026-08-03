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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Observe the symmetry between the original and dual graphs.",
            "Original faces become the dual graph's vertices.",
            "Original vertices become the dual graph's faces.",
            "Both graphs share the same total number of edges.",
            "Euler's formula remains satisfied through this perfect swap."
        ]
        self.setup_layout("Duality and Euler: The Mathematical Swap", lecture_lines)
        
        # Colors
        color_g = BLUE_B
        color_dual = RED_B
        color_swap = YELLOW_A
        color_edge_match = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        # Define Graph G (Tetrahedron projection)
        center_g = (self.grid["B1"] + self.grid["C3"]) / 2 # Moved down slightly from original A1-B3
        v_coords_g = [
            center_g + np.array([0, 0.45, 0]),
            center_g + np.array([-0.4, -0.25, 0]),
            center_g + np.array([0.4, -0.25, 0]),
            center_g + np.array([0, -0.05, 0])
        ]
        
        dots_g = VGroup(*[Dot(p, color=color_g, radius=0.06) for p in v_coords_g])
        edges_g_indices = [(0,1), (1,2), (2,0), (0,3), (1,3), (2,3)]
        edges_g = VGroup(*[Line(v_coords_g[i], v_coords_g[j], color=color_g, stroke_width=3) for i, j in edges_g_indices])
        
        graph_g = VGroup(edges_g, dots_g)
        label_g = Text("Original G", font_size=20, color=color_g)
        self.place_at_grid(label_g, 'A2', scale_factor=0.8) # Issue 33 Fix
        
        # Define Graph G*
        center_dual = (self.grid["B4"] + self.grid["C6"]) / 2 # Moved down slightly
        v_coords_dual = [
            center_dual + np.array([0, -0.4, 0]),
            center_dual + np.array([0.35, 0.2, 0]),
            center_dual + np.array([-0.35, 0.2, 0]),
            center_dual + np.array([0, 0.45, 0])
        ]
        
        dots_dual = VGroup(*[Dot(p, color=color_dual, radius=0.06) for p in v_coords_dual])
        edges_dual_indices = [(0,1), (1,2), (2,0), (0,3), (1,3), (2,3)]
        edges_dual = VGroup(*[Line(v_coords_dual[i], v_coords_dual[j], color=color_dual, stroke_width=3) for i, j in edges_dual_indices])
        
        graph_dual = VGroup(edges_dual, dots_dual)
        label_dual = Text("Dual G*", font_size=20, color=color_dual)
        self.place_at_grid(label_dual, 'A5', scale_factor=0.8) # Issue 33 Fix

        # Table elements
        table_v = Text("V = 4", font_size=24, color=color_g)
        table_e = Text("E = 6", font_size=24, color=color_g)
        table_f = Text("F = 4", font_size=24, color=color_g)
        
        table_v_dual = Text("V* = 4", font_size=24, color=color_dual)
        table_e_dual = Text("E* = 6", font_size=24, color=color_dual)
        table_f_dual = Text("F* = 4", font_size=24, color=color_dual)
        
        self.place_at_grid(table_v, "C2")
        self.place_at_grid(table_e, "D2")
        self.place_at_grid(table_f, "E2")
        
        self.place_at_grid(table_v_dual, "C5")
        self.place_at_grid(table_e_dual, "D5")
        self.place_at_grid(table_f_dual, "E5")
        
        self.lecture[0].set_color(BLUE_B)
        self.play(
            Create(graph_g),
            Write(label_g),
            Write(table_v),
            Write(table_e),
            Write(table_f)
        )
        self.play(
            Create(graph_dual),
            Write(label_dual),
            Write(table_v_dual),
            Write(table_e_dual),
            Write(table_f_dual)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Original faces morph into dual vertices
        self.lecture[1].set_color(YELLOW_A)
        # Faces of G (visualize them before swap)
        faces_g = VGroup(
            Polygon(v_coords_g[1], v_coords_g[2], v_coords_g[3], fill_opacity=0.3, color=color_g, stroke_width=0),
            Polygon(v_coords_g[0], v_coords_g[1], v_coords_g[3], fill_opacity=0.3, color=color_g, stroke_width=0),
            Polygon(v_coords_g[0], v_coords_g[2], v_coords_g[3], fill_opacity=0.3, color=color_g, stroke_width=0)
        )
        arrow_f_v = Arrow(table_f.get_right(), table_v_dual.get_left(), color=color_swap, buff=0.1)
        self.play(FadeIn(faces_g), Indicate(table_f), Indicate(table_v_dual), GrowArrow(arrow_f_v))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Original vertices become dual faces
        self.lecture[2].set_color(BLUE_B)
        arrow_v_f = Arrow(table_v.get_right(), table_f_dual.get_left(), color=color_swap, buff=0.1)
        self.play(FadeOut(faces_g), Indicate(dots_g), Indicate(table_v), Indicate(table_f_dual), GrowArrow(arrow_v_f))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # Same total edges
        self.lecture[3].set_color(GREEN)
        self.play(
            Indicate(table_e),
            Indicate(table_e_dual),
            edges_g.animate.set_color(color_edge_match),
            edges_dual.animate.set_color(color_edge_match)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # Euler's formula
        self.lecture[4].set_color(PINK)
        euler_g = Text("4 - 6 + 4 = 2", font_size=24, color=color_g)
        euler_dual = Text("4 - 6 + 4 = 2", font_size=24, color=color_dual)
        
        # Issue 32 Fix
        self.place_in_area(euler_g, 'F1', 'F3', scale_factor=0.8)
        self.place_in_area(euler_dual, 'F4', 'F6', scale_factor=0.8)
        
        self.play(
            FadeOut(arrow_f_v),
            FadeOut(arrow_v_f),
            Write(euler_g),
            Write(euler_dual)
        )
        self.wait(2)
