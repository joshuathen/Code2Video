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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisites: Continuity and the IVT", ["Continuity connects start to finish.", "The IVT bridges continuous gaps.", "Imagine moving across a room."])
        
        # Room asset
        room = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/room.svg")
        self.place_in_area(room, "A4", "F6", scale_factor=0.6)
        self.add(room)
        
        # === Animation for Lecture Line 1 ===
        # Display a continuous curve inside room
        curve = FunctionGraph(lambda x: 0.5 * np.sin(x * 2), x_range=[-1.5, 1.5], color=WHITE)
        curve.move_to(room.get_center())
        curve_label = Text("Curve", font_size=24, color="#FF0000")
        
        self.play(Create(curve))
        self.place_at_grid(curve_label, "A3", scale_factor=0.8)
        self.play(FadeIn(curve_label))
        self.lecture[0].set_color("#FF0000")

        # === Animation for Lecture Line 2 ===
        # Show the Intermediate Value Theorem visually with a vertical crossing
        line = Line(start=np.array([0, -1, 0]), end=np.array([0, 1, 0]), color="#00FF00")
        self.place_in_area(line, "D4", "F6", scale_factor=0.7)
        self.play(Create(line))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Highlight the intersection point between curve and line
        dot = Dot(color="#FFFF00")
        # Line is at area D4-F6, curve is at center. 
        # Intersection is roughly in the center of the room.
        dot.move_to(curve.point_from_proportion(0.5)) 
        self.play(GrowFromCenter(dot))
        self.lecture[2].set_color("#FFFF00")
        
        self.wait(2)
