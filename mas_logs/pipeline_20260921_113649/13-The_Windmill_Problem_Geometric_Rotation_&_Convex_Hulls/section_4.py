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
        self.setup_layout("Theorem & Result: The Periodicity", [
            "The windmill process is periodic.",
            "It completes cycles of finite rotations.",
            "Total turns relate to point permutations."
        ])
        
        # Elements
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg]
        try:
            windmill_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        except:
            windmill_icon = Circle(radius=0.5, color=WHITE)
            
        blade = Line(start=ORIGIN, end=RIGHT*1.5, color=WHITE)
        pivot = Dot(ORIGIN, color=RED)
        
        windmill_group = VGroup(windmill_icon, blade, pivot)
        # Applying critique: move_in_area C3 to E5
        self.place_in_area(windmill_group, 'C3', 'E5', scale_factor=0.5)
        self.add(windmill_group)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=0.5)
        self.play(Rotate(blade, angle=PI, about_point=windmill_group.get_center()), run_time=3)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"), run_time=0.5)
        # Cycle completion indicator
        cycle_highlight = Arc(radius=1.0, start_angle=0, angle=2*PI, color="#00FFFF")
        cycle_highlight.move_to(windmill_group.get_center())
        self.play(Create(cycle_highlight), run_time=1)
        self.play(FadeOut(cycle_highlight))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"), run_time=0.5)
        # Show periodic function sketch
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 2, 1], axis_config={"include_tip": False}).scale(0.5)
        graph = axes.plot(lambda x: np.sin(2*PI*x) + 1, color="#FF00FF")
        graph_group = VGroup(axes, graph)
        # Applying critique: move_in_area D2 to F5
        self.place_in_area(graph_group, 'D2', 'F5', scale_factor=0.6)
        self.play(Create(axes), Create(graph), run_time=2)
        
        # Repositioning text elements as per critique
        self.place_at_grid(self.lecture, 'A3', scale_factor=0.75)
        
        self.wait(1)
