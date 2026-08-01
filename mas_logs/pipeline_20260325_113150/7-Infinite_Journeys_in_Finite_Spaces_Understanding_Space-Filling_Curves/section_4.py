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
        title = "Critical Properties: Continuity and Locality"
        lecture_lines = [
            "The curve is continuous; the pen never lifts.",
            "Locality means nearby points on the line stay close.",
            "Standard scanning patterns create jumps, breaking this proximity.",
            "This preservation of distance is a unique topological feature.",
            "Continuity and locality define the power of Hilbert curves."
        ]
        self.setup_layout(title, lecture_lines)

        # Hilbert Curve Coordinates (Order 2: 4x4 grid)
        # Normalized to 0-3 range
        hilbert_coords = [
            [0,0], [1,0], [1,1], [0,1], 
            [0,2], [0,3], [1,3], [1,2], 
            [2,2], [2,3], [3,3], [3,2], 
            [3,1], [2,1], [2,0], [3,0]
        ]
        
        # Scaling factor to fit coordinates in the grid system
        scale_val = 0.6
        
        def coords_to_vgroup(coords):
            pts = [np.array([c[0], c[1], 0]) for c in coords]
            vg = VGroup()
            for i in range(len(pts) - 1):
                vg.add(Line(pts[i], pts[i+1], stroke_width=4))
            return vg

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Create the Hilbert Curve
        hilbert_vg = coords_to_vgroup(hilbert_coords)
        self.place_in_area(hilbert_vg, "B2", "E5", scale_factor=0.8)
        
        # Create a trace path for the dot
        trace_path = VMobject(color="#FFFFFF")
        trace_path.set_points_as_corners([l.get_start() for l in hilbert_vg] + [hilbert_vg[-1].get_end()])
        
        dot = Dot(color="#FFFFFF", radius=0.1)
        dot.move_to(trace_path.get_start())
        
        self.add(dot)
        self.play(
            Create(hilbert_vg),
            MoveAlongPath(dot, trace_path),
            run_time=4,
            rate_func=linear
        )
        self.remove(dot)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        
        # Pick two points that are close on the curve (e.g., points 8 and 9)
        p1_pos = hilbert_vg[8].get_start()
        p2_pos = hilbert_vg[8].get_end()
        
        c1 = Circle(radius=0.15, color="#FFFF00").move_to(p1_pos)
        c2 = Circle(radius=0.15, color="#FFFF00").move_to(p2_pos)
        dist_line = Line(p1_pos, p2_pos, color="#FFFF00", stroke_width=6)
        
        self.play(Create(c1), Create(c2))
        self.play(Create(dist_line))
        self.wait(1)
        self.play(FadeOut(c1, c2, dist_line))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        
        # Scanning pattern points (top to bottom, left to right)
        scan_coords = [
            [0,3], [1,3], [2,3], [3,3],
            [0,2], [1,2], [2,2], [3,2],
            [0,1], [1,1], [2,1], [3,1],
            [0,0], [1,0], [2,0], [3,0]
        ]
        
        scan_vg = VGroup()
        for i in range(len(scan_coords)-1):
            l = Line(np.array([scan_coords[i][0], scan_coords[i][1], 0]), 
                     np.array([scan_coords[i+1][0], scan_coords[i+1][1], 0]))
            # Color the "jump" differently
            if scan_coords[i][1] != scan_coords[i+1][1]:
                l.set_color(RED)
            else:
                l.set_color(GRAY)
            scan_vg.add(l)
            
        self.place_in_area(scan_vg, "B2", "E5", scale_factor=0.8)
        
        self.play(hilbert_vg.animate.set_stroke(opacity=0.2))
        self.play(Create(scan_vg), run_time=3)
        
        # Highlight a jump
        jump_line = scan_vg[3] # (3,3) to (0,2)
        jump_label = Text("JUMP", font_size=16, color=RED)
        self.place_at_grid(jump_label, "B4") # Near the jump
        
        self.play(jump_line.animate.set_stroke(width=8), Flash(jump_line, color=RED))
        self.wait(1)
        self.play(FadeOut(scan_vg, jump_label), hilbert_vg.animate.set_stroke(opacity=1))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        
        # Rainbow color gradient along the curve
        rainbow_colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
        hilbert_vg.set_color_by_gradient(*rainbow_colors)
        
        self.play(Indicate(hilbert_vg))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        
        # Final flash and consistent color
        self.play(hilbert_vg.animate.set_color("#00FFFF"))
        self.play(Flash(hilbert_vg, color="#00FFFF", line_length=0.4, num_lines=20))
        self.wait(2)
