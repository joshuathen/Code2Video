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
        self.setup_layout("Visualizing the Linear System", [
            "Represent Ax = b as vector combinations.",
            "Basis vectors a1 and a2 reach b.",
            "Find x and y scaling factors."
        ])
        
        # Setup Axes
        axes = Axes(x_range=[-1, 6, 1], y_range=[-1, 6, 1], x_length=4, y_length=4)
        a1 = Vector([1, 0], color="#FF5733")
        a2 = Vector([0, 1], color="#33FF57")
        b = Vector([5, 5], color=WHITE)
        
        # Applying requested layout adjustments
        self.place_at_grid(axes, 'C3', scale_factor=0.7)
        
        system_group = VGroup(axes, a1, a2, b)
        self.place_in_area(system_group, 'B3', 'E6', scale_factor=0.9)
        
        # Labels
        a1_label = MathTex("a_1", color="#FF5733")
        a2_label = MathTex("a_2", color="#33FF57")
        self.place_at_grid(a1_label, 'E4', scale_factor=0.6)
        self.place_at_grid(a2_label, 'C2', scale_factor=0.6)
        
        # Asset: Slider
        slider1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg")
        slider2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg")
        self.place_at_grid(slider1, 'E2', scale_factor=0.3)
        self.place_at_grid(slider2, 'E5', scale_factor=0.3)

        self.add(axes)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(GrowArrow(a1), GrowArrow(a2), Write(a1_label), Write(a2_label))
        self.play(GrowArrow(b))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(slider1), FadeIn(slider2))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        x_val = ValueTracker(0)
        y_val = ValueTracker(0)
        
        # Creating vectors that track values - keeping it performant
        v1 = Vector([1, 0], color="#FF5733")
        v2 = Vector([0, 1], color="#33FF57")
        axes.add(v1, v2)
        
        v1.add_updater(lambda m: m.become(Vector(x_val.get_value() * np.array([1, 0]), color="#FF5733").move_to(axes.c2p(x_val.get_value()/2, 0))))
        v2.add_updater(lambda m: m.become(Vector(y_val.get_value() * np.array([0, 1]), color="#33FF57").move_to(axes.c2p(0, y_val.get_value()/2))))

        self.play(x_val.animate.set_value(5), y_val.animate.set_value(5), run_time=2)
        self.wait(1)
