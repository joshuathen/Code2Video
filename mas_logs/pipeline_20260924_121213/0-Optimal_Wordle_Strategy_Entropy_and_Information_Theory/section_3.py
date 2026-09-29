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
        lecture_lines = [
            "Greedy strategies pick the most likely word.",
            "Max Entropy picks the best search space reducer.",
            "One eliminates 90%, the other only 10%."
        ]
        self.setup_layout("Algorithmic Analysis: Greedy Search vs. Optimal Path", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Labyrinth asset for search space
        labyrinth = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/labyrinth.svg", color="#8A2BE2")
        self.place_at_grid(labyrinth, "C4", scale_factor=0.6)
        
        root = Circle(radius=0.2, color="#8A2BE2", fill_opacity=1)
        self.place_at_grid(root, "C4", scale_factor=0.9)
        
        tree = VGroup(labyrinth, root)
        
        # Add some child nodes
        for i, pos in enumerate(["D2", "D6"]):
            child = Circle(radius=0.15, color="#8A2BE2", fill_opacity=1)
            self.place_at_grid(child, pos)
            line = Line(root.get_center(), child.get_center(), color=WHITE)
            tree.add(line, child)
            
        self.play(Create(tree))
        self.lecture[0].set_color("#8A2BE2")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Compass asset for orientation
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color="#00FFFF")
        self.place_at_grid(compass, "B6", scale_factor=0.5)

        greedy_path = Line(root.get_center(), self.grid["D2"], color="#00FFFF", stroke_width=4)
        optimum_path = Line(root.get_center(), self.grid["D6"], color="#FFD700", stroke_width=4)
        
        greedy_label = Text("Greedy", font_size=20, color="#00FFFF")
        self.place_at_grid(greedy_label, "D2", scale_factor=0.7)
        
        optimum_label = Text("Optimum", font_size=20, color="#FFD700")
        self.place_at_grid(optimum_label, "D6", scale_factor=0.7)
        
        self.play(Create(greedy_path), Create(optimum_path), Write(greedy_label), Write(optimum_label), FadeIn(compass))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash node selected by algorithm
        flash = Dot(radius=0.3, color="#FF69B4")
        self.place_at_grid(flash, "D5", scale_factor=0.6)
        
        self.play(Flash(flash, color="#FF69B4", line_length=0.2, num_lines=12))
        self.lecture[2].set_color("#FF69B4")
        self.wait(1)
        
        self.play(FadeOut(tree), FadeOut(greedy_path), FadeOut(optimum_path), FadeOut(greedy_label), FadeOut(optimum_label), FadeOut(flash), FadeOut(compass))
