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
        lecture_lines = ["Define boundary between two media.", "Set points A and B.", "Ray intersects boundary at point P.", "Label angles relative to normal.", "Observe path changes with point P."]
        self.setup_layout("Geometric Modeling", lecture_lines)
        
        # Assets
        air = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/air.svg")
        water = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg")
        self.place_at_grid(air, "A5", scale_factor=0.3)
        self.place_at_grid(water, "F5", scale_factor=0.3)

        # Geometric Elements
        boundary = Line(self.grid["C3"], self.grid["C6"], color="#0000FF")
        normal = DashedLine(self.grid["A4"], self.grid["F4"], color=WHITE)
        point_a = Dot(self.grid["B2"], color=YELLOW)
        label_a = Text("A", font_size=20)
        point_b = Dot(self.grid["E5"], color=YELLOW)
        label_b = Text("B", font_size=20)
        point_p = Dot(self.grid["C4"], color=WHITE)
        label_p = Text("P", font_size=20)
        
        ray_in = Line(point_a.get_center(), point_p.get_center(), color=YELLOW)
        ray_out = Line(point_p.get_center(), point_b.get_center(), color=YELLOW)
        
        geometry_group = VGroup(boundary, normal, point_a, point_b, point_p, ray_in, ray_out)

        # === Animation for Lecture Line 1 ===
        self.play(Create(boundary))
        self.lecture[0].set_color("#0000FF")

        # === Animation for Lecture Line 2 ===
        # Use place_at_grid for labels as per Critic instruction
        self.place_at_grid(label_a, 'B3', scale_factor=0.7)
        self.place_at_grid(label_b, 'F5', scale_factor=0.7)
        self.play(Create(point_a), Create(point_b), Write(label_a), Write(label_b))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Position grouping as per Critic instruction
        self.place_in_area(geometry_group, 'C3', 'E5', scale_factor=0.9)
        self.place_at_grid(label_p, 'D4', scale_factor=0.7)
        self.play(Create(normal), Create(point_p), Write(label_p), Create(ray_in), Create(ray_out))
        self.lecture[2].set_color(WHITE)

        # === Animation for Lecture Line 4 ===
        angle_i = Arc(radius=0.5, start_angle=PI/2, angle=-(PI/2 - 0.7), color="#FF0000")
        angle_r = Arc(radius=0.5, start_angle=-PI/2, angle=(PI/2 - 0.5), color="#00FF00")
        angle_arcs = VGroup(angle_i, angle_r)
        self.place_at_grid(angle_arcs, 'D4', scale_factor=0.5)
        self.play(Create(angle_i), Create(angle_r))
        self.lecture[3].set_color("#FF0000")
        
        # === Animation for Lecture Line 5 ===
        new_p_pos = self.grid["C5"]
        self.play(
            point_p.animate.move_to(new_p_pos),
            label_p.animate.move_to(self.grid["D5"]),
            ray_in.animate.put_start_and_end_on(point_a.get_center(), new_p_pos),
            ray_out.animate.put_start_and_end_on(new_p_pos, point_b.get_center()),
            angle_arcs.animate.move_to(new_p_pos)
        )
        self.lecture[4].set_color("#00FF00")
