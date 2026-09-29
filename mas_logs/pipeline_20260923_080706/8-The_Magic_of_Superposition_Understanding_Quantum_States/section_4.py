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
        self.setup_layout("Visualizing Quantum States (Bloch Sphere)", [
            "The Bloch Sphere represents a qubit.",
            "Vectors point anywhere on its surface.",
            "Surface points represent unique state superpositions."
        ])
        
        # --- Visual Elements ---
        # Represent the sphere using the provided asset
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        
        label_0 = Tex(r"$|0\rangle$").scale(0.9)
        label_1 = Tex(r"$|1\rangle$").scale(0.9)
        
        # Use 3D representation if needed, but asset-based for compliance
        sphere_group = VGroup(sphere_icon)
        
        vector = Arrow(start=ORIGIN, end=UP*1.2, color="#FF4500", buff=0)
        
        # --- Animation Stage 1 ---
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#87CEEB")
        
        self.place_at_grid(label_0, 'A3', scale_factor=0.9)
        self.place_at_grid(label_1, 'F3', scale_factor=0.9)
        self.place_in_area(sphere_group, 'B3', 'E5', scale_factor=0.8)
        
        self.play(FadeIn(sphere_group), Write(label_0), Write(label_1))
        
        # --- Animation Stage 2 ---
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF4500")
        
        # Animate vector relative to sphere
        vector.move_to(sphere_group.get_center())
        self.add(vector)
        self.play(Rotate(vector, angle=PI/2, about_point=sphere_group.get_center()))
        
        # --- Animation Stage 3 ---
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFA500")
        
        # Phase indicator using the same asset as context
        arc = Arc(radius=0.7, start_angle=0, angle=PI/4, color="#FFA500")
        arc.move_to(sphere_group.get_center())
        
        self.play(Create(arc))
        self.wait(1)
