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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion & Real-world Application", [
            "Crucial for convex hull algorithms.",
            "State management drives geometric solutions.",
            "Efficiently navigates complex 2D maps."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Summarize algorithm steps
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg]
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        self.place_at_grid(map_icon, 'A4', scale_factor=0.5)
        
        box = SurroundingRectangle(self.lecture[0], buff=0.1, color=WHITE)
        self.play(Create(box))
        
        steps = VGroup(
            Text("1. Sweep Line", font_size=20),
            Text("2. Event Queue", font_size=20),
            Text("3. Update Status", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(steps, 'B1', scale_factor=0.7) # Issue 33 Fix
        
        self.play(FadeIn(steps), FadeIn(map_icon), self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)
        self.play(FadeOut(box))

        # === Animation for Lecture Line 2 ===
        # Show real-world example: collision detection
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png]
        car_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        robot = Circle(radius=0.2, color=YELLOW, fill_opacity=1)
        obstacle = Polygon(LEFT*0.3+UP*0.3, RIGHT*0.3+UP*0.3, RIGHT*0.3+DOWN*0.3, LEFT*0.3+DOWN*0.3, color=RED, fill_opacity=0.5)
        self.place_at_grid(robot, 'C2', scale_factor=0.8) # Issue 32 Fix
        self.place_at_grid(obstacle, 'C4', scale_factor=0.8) # Issue 32 Fix
        self.place_at_grid(car_icon, 'C3', scale_factor=0.2)
        
        path = Line(self.grid['C2'], self.grid['C4'], color=YELLOW)
        
        self.play(FadeIn(robot), FadeIn(obstacle), FadeIn(car_icon))
        self.play(Create(path))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight key takeaways
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot_icon, 'E3', scale_factor=0.5)
        
        takeaways = Text("Geometric Algorithms = State Management", font_size=20, color=GREEN)
        self.place_at_grid(takeaways, 'F2', scale_factor=0.7) # Issue 34 Fix
        
        self.play(Write(takeaways), FadeIn(robot_icon))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
