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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup the layout with specific lines
        lines = [
            'A two-by-three matrix squashes three dimensions into two.',
            'Three basis vectors are flattened onto a flat plane.',
            'Like shadow puppetry, we lose the dimension of depth.',
            'Information is lost when projecting into fewer dimensions.',
            'This transformation reduces the volume into an area.'
        ]
        self.setup_layout("Wide Matrices: The Great Squish (3D to 2D)", lines)

        # Pre-define colors for better visual organization
        colors = ["#6495ED", "#32CD32", "#FFD700", "#FF4500", "#DA70D6"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # Matrix using standard Text
        matrix = Text("M = [[1, 0.5, 0], [0, 0.5, 1]]", font="Monospace", font_size=24, color=WHITE)
        self.place_in_area(matrix, 'A1', 'B6', scale_factor=0.8) # Issue 34
        
        # Simulated 3D Basis Directions
        u_x = np.array([1, -0.15, 0])
        u_y = np.array([0.4, 0.4, 0])
        u_z = np.array([0, 1, 0])
        
        # Basis Vectors created relative to origin
        i_vec = Arrow(ORIGIN, u_x * 1.5, buff=0, color=BLUE)
        j_vec = Arrow(ORIGIN, u_y * 1.5, buff=0, color=GREEN)
        k_vec = Arrow(ORIGIN, u_z * 1.5, buff=0, color=RED)
        
        i_lab = Text("i", slant=ITALIC, color=BLUE, font_size=20).next_to(i_vec.get_end(), RIGHT, buff=0.1)
        j_lab = Text("j", slant=ITALIC, color=GREEN, font_size=20).next_to(j_vec.get_end(), UR, buff=0.1)
        label_k = Text("k", slant=ITALIC, color=RED, font_size=20)
        
        # Group and position
        vector_projection = VGroup(i_vec, j_vec, k_vec, i_lab, j_lab, label_k)
        self.place_in_area(vector_projection, 'C2', 'F5', scale_factor=1.1) # Issue 35
        self.place_at_grid(label_k, 'C4', scale_factor=0.7) # Issue 36
        
        # Captured origin after placement
        origin = i_vec.get_start()

        self.play(Write(matrix))
        self.play(Create(vector_projection))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        
        # Target 2D vectors based on matrix mapping: i->(1,0), j->(0.5,0.5), k->(0,1)
        t_i = Arrow(origin, origin + RIGHT * 1.2, buff=0, color=BLUE)
        t_j = Arrow(origin, origin + (RIGHT + UP) * 0.6, buff=0, color=GREEN)
        t_k = Arrow(origin, origin + UP * 1.2, buff=0, color=RED)
        
        self.play(
            Transform(i_vec, t_i),
            Transform(j_vec, t_j),
            Transform(k_vec, t_k),
            i_lab.animate.next_to(t_i.get_end(), RIGHT, buff=0.1),
            j_lab.animate.next_to(t_j.get_end(), UR, buff=0.1),
            label_k.animate.next_to(t_k.get_end(), UP, buff=0.1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        def get_cube(origin_pt, bx, by, bz, scale=0.7, color=WHITE):
            pts = [np.array([x,y,z]) for x in [0,1] for y in [0,1] for z in [0,1]]
            edges = []
            for i in range(8):
                for j in range(i+1, 8):
                    if np.sum(np.abs(pts[i]-pts[j])) == 1:
                        p1 = origin_pt + (pts[i][0]*bx + pts[i][1]*by + pts[i][2]*bz)*scale
                        p2 = origin_pt + (pts[j][0]*bx + pts[j][1]*by + pts[j][2]*bz)*scale
                        edges.append(Line(p1, p2, stroke_width=2, color=color))
            return VGroup(*edges)

        cube_sim_3d = get_cube(origin + DOWN*0.5 + LEFT*0.5, u_x, u_y, u_z)
        self.play(Create(cube_sim_3d))
        
        shadow = get_cube(origin + DOWN*0.5 + LEFT*0.5, RIGHT, (RIGHT+UP)*0.5, UP, color="#FFFF00")
        for edge in shadow: edge.set_stroke(width=4)
        
        self.play(Transform(cube_sim_3d.copy(), shadow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        
        # P1 and P2 map to same shadow point under M
        # Point 1: [1, 1, 1], Point 2: [1.5, 0, 1.5]
        # Both project to [1.5, 1.5]
        p1_coord = np.array([1, 1, 1])
        p2_coord = np.array([1.5, 0, 1.5])
        
        p1_sim = origin + (p1_coord[0]*u_x + p1_coord[1]*u_y + p1_coord[2]*u_z)*0.6
        p2_sim = origin + (p2_coord[0]*u_x + p2_coord[1]*u_y + p2_coord[2]*u_z)*0.6
        p_out = origin + (1.5*RIGHT + 1.5*UP)*0.6
        
        dot1 = Dot(p1_sim, color=BLUE, radius=0.07)
        dot2 = Dot(p2_sim, color=ORANGE, radius=0.07)
        dot_out = Dot(p_out, color=YELLOW, radius=0.09)
        
        self.play(FadeIn(dot1), FadeIn(dot2))
        self.play(
            dot1.animate.move_to(p_out),
            dot2.animate.move_to(p_out),
            FadeIn(dot_out)
        )
        
        loss_text = Text("3D Depth lost in shadow", font_size=16, color=YELLOW)
        self.place_at_grid(loss_text, "F4")
        self.play(Write(loss_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        
        output_space_label = Text("Output Space", color=WHITE, font_size=24)
        self.place_at_grid(output_space_label, "F6", scale_factor=1.0)
        self.play(Write(output_space_label))
        self.wait(2)
