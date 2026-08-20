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
        lecture_lines = [
            "Position graphs transform into velocity graphs.",
            "The derivative measures instantaneous growth factors.",
            "A cheetah's speed is the curve's steepness.",
            "The slope represents speed at a micro-second.",
            "Physics relies on these dynamic rates."
        ]
        self.setup_layout("Dynamic Application: Rate of Change", lecture_lines)
        
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], axis_config={"include_tip": True})
        func = lambda t: 0.2 * t**2
        graph = axes.plot(func, color=YELLOW)
        
        # Cheetah icon
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        
        # Fix 30: Adjust axes position
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        # Asset integration: cheetah
        cheetah.set_color(YELLOW)
        self.place_at_grid(cheetah, 'B4', scale_factor=0.3)
        self.play(Create(axes), Create(graph), FadeIn(cheetah))
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        # Fix 32: Positioning the point
        point = Dot(color=ORANGE)
        self.place_at_grid(point, 'D4', scale_factor=0.6)
        self.play(FadeIn(point))
        self.play(self.lecture[1].animate.set_color(ORANGE))

        # === Animation for Lecture Line 3 ===
        tangent_line = TangentLine(graph, alpha=0.5, length=2, color=RED)
        self.play(Create(tangent_line))
        self.play(self.lecture[2].animate.set_color(RED))

        # === Animation for Lecture Line 4 ===
        # Fix 31: Positioning the formula
        slope_label = MathTex("m = f'(t)").set_color(BLUE)
        self.place_at_grid(slope_label, 'E3', scale_factor=0.7)
        self.play(Write(slope_label))
        
        # Asset integration: cheetah
        cheetah_speed = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        cheetah_speed.set_color(BLUE)
        self.place_at_grid(cheetah_speed, 'E4', scale_factor=0.3)
        self.play(FadeIn(cheetah_speed))
        
        self.play(self.lecture[3].animate.set_color(BLUE))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GREEN))
        self.wait(2)
