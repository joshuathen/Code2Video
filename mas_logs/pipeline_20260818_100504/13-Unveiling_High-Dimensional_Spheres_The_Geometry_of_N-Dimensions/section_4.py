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
        lecture_lines = ["We use orthogonal projections to visualize 4D objects.", "A 4D sphere slices through our 3D reality.", "It appears, grows, peaks, and then vanishes."]
        self.setup_layout("Visualizing the Un-visualizable", lecture_lines)
        
        # Elements for animation
        # Replacing sphere with SVG as per instructions
        sphere_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        sphere = SVGMobject(sphere_asset)
        label = Text("Hypersphere", font_size=32)
        self.place_at_grid(label, "B3", scale_factor=1.2)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        
        # Animation: sphere appears/grows
        # Updated positioning based on VideoCritic
        self.place_in_area(sphere, "D4", "E5", scale_factor=0.4)
        sphere.set_color("#33FF57")
        self.play(FadeIn(sphere))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        
        # Animation: flash label
        self.play(Flash(label.get_center(), color="#3357FF", line_length=0.2))
        
        # Animation: sphere scales/vanishes
        self.play(sphere.animate.set_color("#3357FF").scale(1.5))
        self.play(FadeOut(sphere))
        self.wait(2)
