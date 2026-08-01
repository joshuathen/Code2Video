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
        self.setup_layout(
            "Puzzle 1: The Shortest Path (3D to 2D Shift)",
            [
                "A spider wants the shortest path to a fly.",
                "Navigating a 3D room makes the path look complex.",
                "We can flatten the room into a single 2D plane.",
                "On this flat map, the shortest path is a line.",
                "This 2D shift reveals the hidden hypotenuse path."
            ]
        )

        # Assets
        spider_svg_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/spider.svg"

        # === Animation for Lecture Line 1 ===
        # Show a 3D rectangular box in grey #808080 with dots for spider and fly.
        
        # Room geometry (3D projection)
        front_wall = Square(side_length=1.5, color="#808080", fill_opacity=0.3)
        side_wall = Polygon(
            front_wall.get_corner(UR),
            front_wall.get_corner(UR) + RIGHT * 1 + UP * 0.5,
            front_wall.get_corner(DR) + RIGHT * 1 + UP * 0.5,
            front_wall.get_corner(DR),
            color="#808080", fill_opacity=0.3
        )
        
        room_net_group = VGroup(front_wall, side_wall)
        self.place_in_area(room_net_group, 'B2', 'E5', scale_factor=0.9)
        
        # Spider (SVG) and Fly (Dot)
        spider = SVGMobject(spider_svg_path, color=WHITE).scale(0.15)
        spider.move_to(front_wall.get_corner(DL))
        
        fly = Dot(color=RED, radius=0.08)
        fly.move_to(side_wall.get_vertices()[1]) # Top-right vertex of the side wall
        
        self.lecture[0].set_color(YELLOW)
        self.play(Create(room_net_group), FadeIn(spider), FadeIn(fly), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A dashed pink #FF00FF line shows the spider's path across two walls.
        
        # Intersection point on the shared edge between the two walls
        shared_edge_point = (front_wall.get_corner(UR) + front_wall.get_corner(DR)) / 2 + DOWN * 0.2
        path_seg1 = DashedLine(spider.get_center(), shared_edge_point, color="#FF00FF")
        path_seg2 = DashedLine(shared_edge_point, fly.get_center(), color="#FF00FF")
        complex_path = VGroup(path_seg1, path_seg2)
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(complex_path), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The box walls unfold to form a flat 2D plane.
        
        # Flattened net geometry
        flat_front = Square(side_length=1.5, color="#808080", fill_opacity=0.3)
        flat_side = Square(side_length=1.5, color="#808080", fill_opacity=0.3).next_to(flat_front, RIGHT, buff=0)
        net_2d_group = VGroup(flat_front, flat_side)
        self.place_in_area(net_2d_group, 'B2', 'E5', scale_factor=0.9)
        
        # Target positions on the flat net
        spider_target_pos = flat_front.get_corner(DL)
        fly_target_pos = flat_side.get_corner(UR)
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(
            FadeOut(complex_path),
            ReplacementTransform(front_wall, flat_front),
            ReplacementTransform(side_wall, flat_side),
            spider.animate.move_to(spider_target_pos),
            fly.animate.move_to(fly_target_pos),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The spider and fly dots are now positioned on the flat plane.
        
        net_label = Text("Unfolded Net", font_size=20, color="#808080")
        self.place_at_grid(net_label, 'F3', scale_factor=0.8)
        
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(FadeIn(net_label), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # A solid green #00FF00 line connects the dots, showing the shortest path.
        
        shortest_path = Line(spider.get_center(), fly.get_center(), color="#00FF00", stroke_width=4)
        hypotenuse_label = Text("Hypotenuse", font_size=18, color="#00FF00")
        self.place_at_grid(hypotenuse_label, 'B4', scale_factor=0.6)
        
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(Create(shortest_path), FadeIn(hypotenuse_label), run_time=2)
        self.wait(2)
        self.lecture[4].set_color(WHITE)
