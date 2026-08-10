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
            "Fix error by propagating it backward through layers.",
            "Calculate contribution of each weight to total error.",
            "Use the chain rule like cascading levers.",
            "The chef blames the cook to adjust the recipe.",
            "Backward adjustments refine the network's future performance."
        ]
        self.setup_layout("Backpropagation: Blaming the Weights", lecture_lines)
        
        # Create visual elements
        net = VGroup()
        for i in range(3):
            col = VGroup(*[Circle(radius=0.2, color=BLUE) for _ in range(3)]).arrange(DOWN, buff=0.4)
            net.add(col)
        net.arrange(RIGHT, buff=0.8)
        
        # Applying layout fixes per VideoCritic instructions
        self.place_in_area(net, "B3", "E5", scale_factor=0.65)
        
        # Error indicator
        error_label = MathTex(r"E", color=RED)
        self.place_at_grid(error_label, "D5", scale_factor=0.9)
        
        # Load assets
        chef_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chef.svg")
        cook_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cook.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(chef_icon, "A5", scale_factor=0.8)
        self.play(FadeIn(chef_icon), Indicate(error_label, color=RED))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Indicate(net[2]))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        levers = VGroup(*[Line(net[i].get_center(), net[i+1].get_center(), color=GRAY) for i in range(2)])
        self.play(Create(levers))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        path = Line(net[2].get_center(), net[0].get_center(), color=RED, stroke_width=4)
        self.play(Create(path))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.place_at_grid(cook_icon, "B2", scale_factor=0.8)
        self.play(FadeIn(cook_icon), FadeOut(path), FadeOut(levers), net.animate.set_color(GREEN))
