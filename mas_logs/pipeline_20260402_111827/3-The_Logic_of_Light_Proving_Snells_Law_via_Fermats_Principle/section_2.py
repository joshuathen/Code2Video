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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite: Fermat's Principle of Least Time", 
            [
                "Fermat’s Principle states light takes the quickest path.",
                "Index n equals vacuum speed c divided by material speed.",
                "Light optimizes its journey between any two points."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Define Points and Objects
        # Issue 35: Point A anchored to grid
        point_a = Dot(color=WHITE)
        self.place_at_grid(point_a, 'B2', scale_factor=0.8)
        label_a = Text("A", font_size=24, color=WHITE)
        self.place_at_grid(label_a, 'A2', scale_factor=1.0)
        
        # Issue 36: Point B anchored to grid
        point_b = Dot(color=WHITE)
        self.place_at_grid(point_b, 'E5', scale_factor=0.8)
        label_b = Text("B", font_size=24, color=WHITE)
        self.place_at_grid(label_b, 'F5', scale_factor=1.0)
        
        # Issue 37: Light path connecting the two points
        light_path = Line(self.grid['B2'], self.grid['E5'], color="#FFFF00")
        self.place_in_area(light_path, 'B2', 'E5', scale_factor=1.0)
        
        self.play(FadeIn(point_a), FadeIn(label_a), FadeIn(point_b), FadeIn(label_b))
        self.play(Create(light_path))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Horizontal boundary at row D
        boundary = Line(self.grid['D1'], self.grid['D6'], color=BLUE_B)
        boundary_label = Text("Boundary", font_size=20, color=BLUE_B)
        self.place_at_grid(boundary_label, 'C1', scale_factor=1.0)
        
        # Path Candidates: Straight (already visible), Early, and Late
        # Straight path crosses boundary at D4 approximately
        # Early path: hits boundary at D3
        path_early = VMobject(color=WHITE).set_points_as_corners([self.grid['B2'], self.grid['D3'], self.grid['E5']])
        # Late path: hits boundary at D5
        path_late = VMobject(color=WHITE).set_points_as_corners([self.grid['B2'], self.grid['D5'], self.grid['E5']])
        
        self.play(Create(boundary), FadeIn(boundary_label))
        self.play(Create(path_early), Create(path_late))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Fermat's principle optimization
        # Optimal path (least time) highlighted in Gold
        pos_inter = (self.grid['D4'] + self.grid['D5']) / 2 # Slightly bent
        least_time_path = VMobject(color="#FFD700").set_points_as_corners([self.grid['B2'], pos_inter, self.grid['E5']])
        least_time_path.set_stroke(width=6)
        
        formula_n = Text("n = c/v", font_size=24, color="#FFFFFF")
        self.place_at_grid(formula_n, 'F6', scale_factor=1.0)
        
        self.play(
            FadeOut(path_early), 
            FadeOut(path_late), 
            FadeOut(light_path)
        )
        self.play(Create(least_time_path))
        self.play(Write(formula_n))
        self.wait(2)
