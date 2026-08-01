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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup title and script
        title = "The Net Puzzle: Unfolding the Third Dimension"
        lines = [
            'A 3D cube can unfold into a net.',
            'Boxy needs to cross this 3D room.',
            'Unfolding the walls reveals a 2D shortcut.',
            'The shortest path is a straight 2D line.',
            'Folding back creates a clever 3D diagonal.'
        ]
        self.setup_layout(title, lines)

        # Colors
        CUBE_COLOR = "#A9A9A9"
        NET_COLOR = "#DDA0DD"
        PATH_COLOR = "#00FF00"
        HIGHLIGHT_COLOR = YELLOW

        # --- Assets Construction ---
        
        # 1. Cube Wireframe (2D Projection)
        s = 1.0 # side length
        o = 0.4 # offset for 3D look
        
        front_pts = [np.array([0,0,0]), np.array([s,0,0]), np.array([s,s,0]), np.array([0,s,0])]
        back_pts = [p + np.array([o,o,0]) for p in front_pts]
        
        cube_lines = VGroup()
        # Front face
        for i in range(4):
            cube_lines.add(Line(front_pts[i], front_pts[(i+1)%4], color=CUBE_COLOR))
        # Back face
        for i in range(4):
            cube_lines.add(Line(back_pts[i], back_pts[(i+1)%4], color=CUBE_COLOR))
        # Connecting edges
        for i in range(4):
            cube_lines.add(Line(front_pts[i], back_pts[i], color=CUBE_COLOR))
            
        cube_group = cube_lines.copy()
        # Adjusted scale from 1.2 to 1.0 to fit area better (Issue 34)
        self.place_in_area(cube_group, "B2", "E5", scale_factor=1.0)
        
        # 2. Points A and B on Cube
        # Let's define A as the Front-Bottom-Left (index 0) and B as Back-Top-Right (index 6)
        dot_a = Dot(cube_group[8].get_start(), color=WHITE) # Back-Bottom-Left equivalent in the logic
        dot_b = Dot(cube_group[6].get_end(), color=WHITE)   # Back-Top-Right
        
        # Labeling relative to the group
        label_a = Text("A", font_size=16).next_to(dot_a, LEFT, buff=0.1)
        label_b = Text("B", font_size=16).next_to(dot_b, RIGHT, buff=0.1)
        labels = VGroup(label_a, label_b)

        # 3. The Net (T-shape)
        # Using a standard cross net:
        #    [T]
        # [L][F][R]
        #    [Bm]
        #    [Bk]
        sq_s = 0.7
        net_squares = VGroup()
        positions = [
            [0, 0, 0],    # F (Center)
            [-sq_s, 0, 0], # L
            [sq_s, 0, 0],  # R
            [0, sq_s, 0],  # T
            [0, -sq_s, 0], # Bm
            [0, -2*sq_s, 0] # Bk
        ]
        for pos in positions:
            sq = Square(side_length=sq_s, color=NET_COLOR).move_to(pos)
            net_squares.add(sq)
        
        net_group = net_squares.copy()
        # Adjusted scale from 1.0 to 0.8 to fit area better (Issue 35)
        self.place_in_area(net_group, "B2", "E5", scale_factor=0.8)
        
        # 4. Path on Net
        # If A is top-left of Left face and B is bottom-right of Back face
        # Square indices: F=0, L=1, R=2, T=3, Bm=4, Bk=5
        # Top-left of L: net_group[1].get_corner(UL)
        # Bottom-right of Bk: net_group[5].get_corner(DR)
        path_start = net_group[1].get_corner(UL)
        path_end = net_group[5].get_corner(DR)
        net_path = Line(path_start, path_end, color=PATH_COLOR, stroke_width=4)

        # 5. Path on Cube (Approximate folding)
        # The path crosses L, F, Bm, Bk (based on our net layout)
        # In 3D wireframe, this is a sequence of lines.
        cube_path = VGroup(
            Line(cube_group[11].get_end(), cube_group[0].get_start(), color=PATH_COLOR), # Placeholder segments
            Line(cube_group[0].get_start(), cube_group[1].get_end(), color=PATH_COLOR)
        )
        # We will just use a dashed polyline for the 'diagonal' effect on the cube
        cube_path_final = DashedLine(dot_a.get_center(), dot_b.get_center(), color=PATH_COLOR)

        # === Animation for Lecture Line 1 ===
        # 'A 3D cube can unfold into a net.'
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.play(Create(cube_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 'Boxy needs to cross this 3D room.'
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        self.play(FadeIn(dot_a), FadeIn(dot_b), Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 'Unfolding the walls reveals a 2D shortcut.'
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        # Transform cube + points to net
        # We'll transform labels and points to the net corners
        dot_a_net = Dot(path_start, color=WHITE)
        dot_b_net = Dot(path_end, color=WHITE)
        label_a_net = Text("A", font_size=16).next_to(dot_a_net, LEFT, buff=0.1)
        label_b_net = Text("B", font_size=16).next_to(dot_b_net, RIGHT, buff=0.1)
        
        self.play(
            ReplacementTransform(cube_group, net_group),
            ReplacementTransform(dot_a, dot_a_net),
            ReplacementTransform(dot_b, dot_b_net),
            ReplacementTransform(label_a, label_a_net),
            ReplacementTransform(label_b, label_b_net),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # 'The shortest path is a straight 2D line.'
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        self.play(Create(net_path))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # 'Folding back creates a clever 3D diagonal.'
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Redefine cube group for transform back
        cube_group_final = cube_lines.copy()
        # Adjusted scale from 1.2 to 1.0 to match original and maintain consistency (Issue 36)
        self.place_in_area(cube_group_final, "B2", "E5", scale_factor=1.0)
        dot_a_final = Dot(cube_group_final[8].get_start(), color=WHITE)
        dot_b_final = Dot(cube_group_final[6].get_end(), color=WHITE)
        label_a_final = Text("A", font_size=16).next_to(dot_a_final, LEFT, buff=0.1)
        label_b_final = Text("B", font_size=16).next_to(dot_b_final, RIGHT, buff=0.1)
        
        self.play(
            ReplacementTransform(net_group, cube_group_final),
            ReplacementTransform(dot_a_net, dot_a_final),
            ReplacementTransform(dot_b_net, dot_b_final),
            ReplacementTransform(label_a_net, label_a_final),
            ReplacementTransform(label_b_net, label_b_final),
            ReplacementTransform(net_path, cube_path_final),
            run_time=2
        )
        self.wait(2)
        self.lecture[4].set_color(WHITE)
