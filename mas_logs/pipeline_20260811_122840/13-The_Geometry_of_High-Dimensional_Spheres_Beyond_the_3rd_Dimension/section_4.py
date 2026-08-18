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
        self.setup_layout("Visualizing the Corners", [
            "Spheres reside inside hypercubes.",
            "Corners stretch far from the center.",
            "Distance to corners grows exponentially."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display 2D square corners, labeled A, B, C, D in #FFFFFF.
        sq = Square(side_length=2, color=WHITE)
        self.place_in_area(sq, "B4", "E6", scale_factor=0.5)
        
        labels = VGroup(*[Text(l, font_size=24, color=WHITE) for l in ["A", "B", "C", "D"]])
        # Tether labels using next_to as per B011
        for i, pos in enumerate(["B4", "B6", "E6", "E4"]):
            self.place_at_grid(labels[i], pos, scale_factor=0.7)
            labels[i].next_to(sq, {"B4": UL, "B6": UR, "E6": DR, "E4": DL}[pos], buff=0.1)
        
        self.play(Create(sq), Write(labels))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Add dimension 3 with 8 corners represented by sphere.svg, flashing #FFFF00.
        cube = Cube(side_length=2, fill_opacity=0.3, stroke_color=WHITE)
        self.place_in_area(cube, "B4", "E6", scale_factor=0.5)
        
        sphere_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        corners = VGroup(*[SVGMobject(sphere_asset, color="#FFFF00") for _ in range(8)])
        
        # Position corners roughly around cube
        pos = [
            [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1],
            [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]
        ]
        for i, dot in enumerate(corners):
            dot.scale(0.15)
            dot.move_to(cube.get_center() + np.array(pos[i]) * 0.7)
            
        self.play(ReplacementTransform(sq, cube), FadeOut(labels), FadeIn(corners))
        self.play(Flash(corners, color="#FFFF00", flash_radius=0.3))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show exponential corner growth using multiple sphere.svg, color #00FFFF.
        self.play(corners.animate.set_color("#00FFFF"))
        
        # Add extra spheres to show density/growth
        more_spheres = VGroup(*[SVGMobject(sphere_asset, color="#00FFFF").scale(0.1) for _ in range(8)])
        for m in more_spheres:
            m.move_to(cube.get_center() + np.random.uniform(-1.5, 1.5, 3))
        
        self.play(FadeIn(more_spheres))
        
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
