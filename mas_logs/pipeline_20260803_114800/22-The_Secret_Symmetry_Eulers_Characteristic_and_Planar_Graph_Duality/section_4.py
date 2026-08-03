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
        # Data from storyboard
        title = "The Symmetry Swap: Relationships between Primal and Dual"
        lines = [
            "Dual vertices equal the number of original faces.",
            "Dual faces equal the number of original vertices.",
            "Both graphs share the same number of edges.",
            "Euler's formula holds true for both versions.",
            "Some special graphs are even self-dual."
        ]
        self.setup_layout(title, lines)

        # Colors
        PRIMAL_COLOR = "#FFFFFF"
        DUAL_COLOR = "#FF00FF"
        HIGHLIGHT_COLOR = "#FFFF00"
        SELF_DUAL_COLOR = "#00FF00"

        # Graphs setup
        # Primal: Square with diagonal (V=4, E=5, F=3)
        p_coords = [np.array([-1,1,0]), np.array([1,1,0]), np.array([1,-1,0]), np.array([-1,-1,0])]
        p_v = VGroup(*[Dot(c, color=PRIMAL_COLOR) for c in p_coords])
        p_e_idx = [(0,1), (1,2), (2,3), (3,0), (0,2)]
        p_e = VGroup(*[Line(p_coords[i], p_coords[j], color=PRIMAL_COLOR) for i,j in p_e_idx])
        # Define internal faces for highlighting
        p_f1 = Polygon(p_coords[0], p_coords[1], p_coords[2], fill_opacity=0, stroke_width=0, color=HIGHLIGHT_COLOR)
        p_f2 = Polygon(p_coords[0], p_coords[2], p_coords[3], fill_opacity=0, stroke_width=0, color=HIGHLIGHT_COLOR)
        primal_graph = VGroup(p_f1, p_f2, p_e, p_v)

        # Dual: Multigraph for Square+Diagonal
        d_coords = [
            np.array([0.4, 0.4, 0]),   # Center of f1
            np.array([-0.4, -0.4, 0]), # Center of f2
            np.array([1.5, 0, 0])      # Point in exterior face
        ]
        d_v = VGroup(*[Dot(c, color=DUAL_COLOR) for c in d_coords])
        d_e = VGroup(
            CurvedArrow(d_coords[0], d_coords[2], angle=TAU/4, color=DUAL_COLOR, tip_length=0),
            CurvedArrow(d_coords[0], d_coords[2], angle=-TAU/4, color=DUAL_COLOR, tip_length=0),
            CurvedArrow(d_coords[1], d_coords[2], angle=TAU/4, color=DUAL_COLOR, tip_length=0),
            CurvedArrow(d_coords[1], d_coords[2], angle=-TAU/4, color=DUAL_COLOR, tip_length=0),
            Line(d_coords[0], d_coords[1], color=DUAL_COLOR)
        )
        dual_graph = VGroup(d_e, d_v)

        # Positioning
        self.place_in_area(primal_graph, "B1", "E3", scale_factor=0.7)
        self.place_in_area(dual_graph, "B4", "E6", scale_factor=0.7)
        
        primal_label = Text("Primal", font_size=24, color=PRIMAL_COLOR)
        dual_label = Text("Dual", font_size=24, color=DUAL_COLOR)
        self.place_at_grid(primal_label, "A2")
        self.place_at_grid(dual_label, "A5")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(PRIMAL_COLOR))
        self.play(Create(primal_graph), Create(dual_graph), Write(primal_label), Write(dual_label))
        
        v_star_f = Text("V* = F", font_size=24, color=WHITE)
        self.place_at_grid(v_star_f, "F3")
        
        self.play(Write(v_star_f))
        # Highlight: Dual vertices correspond to Primal faces
        self.play(
            p_f1.animate.set_fill(opacity=0.4), 
            p_f2.animate.set_fill(opacity=0.4), 
            Indicate(d_v)
        )
        self.wait(1)
        self.play(p_f1.animate.set_fill(opacity=0), p_f2.animate.set_fill(opacity=0))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(PRIMAL_COLOR))
        f_star_v = Text("F* = V", font_size=24, color=WHITE)
        self.place_at_grid(f_star_v, "F4")
        
        self.play(Write(f_star_v))
        # Dual faces correspond to Primal vertices
        self.play(Indicate(p_v))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(PRIMAL_COLOR))
        e_star_e = Text("E* = E", font_size=24, color=WHITE)
        self.place_at_grid(e_star_e, "F5")
        
        self.play(Write(e_star_e))
        # Edge sets are identical in size
        self.play(Indicate(p_e), Indicate(d_e))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(PRIMAL_COLOR))
        euler_p = Text("V-E+F=2", font_size=20, color=PRIMAL_COLOR)
        euler_d = Text("V*-E*+F*=2", font_size=20, color=DUAL_COLOR)
        # Fix: Positioning of Euler Primal formula
        self.place_at_grid(euler_p, "F2")
        self.place_at_grid(euler_d, "F6")
        
        self.play(Write(euler_p), Write(euler_d))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(SELF_DUAL_COLOR))
        
        # Transition to Self-Dual example (Tetrahedron)
        self.play(
            FadeOut(primal_graph), FadeOut(dual_graph), 
            FadeOut(primal_label), FadeOut(dual_label),
            FadeOut(v_star_f), FadeOut(f_star_v), FadeOut(e_star_e), 
            FadeOut(euler_p), FadeOut(euler_d)
        )
        
        # Tetrahedron Planar representation
        t_coords = [
            np.array([0, 1.2, 0]), 
            np.array([-1.2, -0.8, 0]), 
            np.array([1.2, -0.8, 0]), 
            np.array([0, 0, 0])
        ]
        t_v = VGroup(*[Dot(c, color=SELF_DUAL_COLOR) for c in t_coords])
        t_e_idx = [(0,1), (1,2), (2,0), (0,3), (1,3), (2,3)]
        t_e = VGroup(*[Line(t_coords[i], t_coords[j], color=SELF_DUAL_COLOR) for i,j in t_e_idx])
        tet_graph = VGroup(t_e, t_v)
        
        self.place_in_area(tet_graph, "B2", "E5", scale_factor=0.9)
        
        sd_label = Text("Self-Dual: Tetrahedron", color=SELF_DUAL_COLOR, font_size=24)
        # Fix: Centering of self-dual label
        self.place_at_grid(sd_label, "A4")
        
        self.play(Create(tet_graph), Write(sd_label))
        self.wait(2)
